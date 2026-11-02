#!/usr/bin/env python3
import os
import math
import time
import random
import json
import shutil
import webbrowser
import threading
from datetime import datetime
import shapely
import csv

import geopandas as gpd
import pandas as pd

from coverage_app.flask_routes import configure_scenario_files, run_flask_app, stop_flask_app
from coverage_app.coverage_analysis import combine_scenario_antennas
from coverage_app.coverage_analysis import generate_coverage_points
from coverage_app.coverage_analysis import perform_coverage_analysis
from coverage_app.path_loss import choose_path_loss_model
from coverage_app.optimization import optimize_scenario
from coverage_app.cli_utils import (
    _banner, _section, _info, _ok, _warn, _err, _wait, _ask, _ask_int, _ask_float,
    RST, BOLD, DIM, CYAN, GREEN, YELLOW, RED,
)
from coverage_app.fetch import fetch_geodata

def _shutdown_flask():
        try:
                if stop_flask_app(timeout=5):
                    _ok("Flask server stopped.")
                else:
                    _warn("Flask shutdown may not have completed.")
        except Exception as e:
            _warn(f"Shutdown error: {e}")

def _collect_vstation_params(polygons_gdf, real_antennas, default_label=""):
    """Interactively collect virtual station generation parameters."""
    _section(f"Virtual Antenna Configuration{default_label}")
    nstations        = _ask_int("Number of virtual base stations", 2)
    nants_per_station = _ask_int("Antennas per station",           2)
    max_radius       = _ask_float("Max radius from AOI center (m)", 2000)
    ptx_min          = _ask_float("Min Tx power – Ptx (dBm)",       40)
    ptx_max          = _ask_float("Max Tx power – Ptx (dBm)",       50)
    gtx_min          = _ask_float("Min Tx gain  – Gtx (dBi)",       15)
    gtx_max          = _ask_float("Max Tx gain  – Gtx (dBi)",       20)

    if len(polygons_gdf) > 0:
        c = polygons_gdf.iloc[0].geometry.centroid
        c_lat, c_lon = c.y, c.x
    elif real_antennas:
        c_lat, c_lon = real_antennas[0]["lat"], real_antennas[0]["lon"]
    else:
        c_lat, c_lon = 42.90076, -78.8650
        return nstations, nants_per_station, max_radius, (ptx_min, ptx_max), (gtx_min, gtx_max), c_lat, c_lon

def list_geojson_files():
    """
        Return a list of .geojson files in the same directory as main.py
        (which is also where geojson_creator saves scenario.geojson).
        """
    search_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    for f in os.listdir(search_dir):
        if f.lower().endswith(".geojson"):
            candidates.append(os.path.join(search_dir, f))
    return candidates, search_dir

def load_geojson_as_gdf(geojson_path):
    """
        Loads a .geojson file into two GeoDataFrames: 
        - polygons_gdf with Polygon or MultiPolygon
        - points_gdf with Point
        Ensures CRS = EPSG:4326.
        """
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
        
    polygons = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    points = gdf[gdf.geometry.type == "Point"]
    if len(polygons) > 0:
        poly_union = polygons.union_all()
        polygons_gdf = gpd.GeoDataFrame(geometry=[poly_union], crs="EPSG:4326")
    else:
        polygons_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return polygons_gdf, points

def generate_vstations(num_stations, num_ants_per_stn, radius_m, ptx_rng, gtx_rng, c_lat, c_lon, freq_list):
    """
        Generates random base stations around (c_lat, c_lon) within 'radius_m'.
        Each station has 'num_ants_per_stn' antennas, randomizing azimuth, Ptx, Gtx.
        """
    vants = []
    for _ in range(num_stations):
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(0, radius_m)
        dlat = (dist * math.cos(angle)) / 111320
        dlon = (dist * math.sin(angle)) / (111320 * math.cos(math.radians(c_lat)))
        st_lat = round(c_lat + dlat, 3)
        st_lon = round(c_lon + dlon, 3)

        for _a in range(num_ants_per_stn):
            freq_ = random.choice(freq_list)
            ant_ = {
                "lat": st_lat,
                "lon": st_lon,
                "azimuth": round(random.uniform(0, 360), 2),
                "frequency": freq_,
                "Ptx": round(random.uniform(ptx_rng[0], ptx_rng[1]), 2),
                "Gtx": round(random.uniform(gtx_rng[0], gtx_rng[1]), 2),
                "height": round(random.uniform(20, 40), 2),
                "type": "virtual"
            }
            vants.append(ant_)
    return vants


def _slugify_name(value):
    cleaned = []
    for char in str(value).strip().lower():
            if char.isalnum() or char in ("-", "_"):
                cleaned.append(char)
            elif char.isspace() or char in (".", "/", "\\"):
                cleaned.append("_")
            else:
                cleaned.append("_")
    result = "".join(cleaned).strip("_")
    while "__" in result:
        result = result.replace("__", "_")
    return result or "scenario"


def _build_coverage_run_layout(script_dir, scenario_path, run_label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_name = os.path.splitext(os.path.basename(scenario_path))[0] if scenario_path else "scenario"
    scenario_slug = _slugify_name(scenario_name)
    run_slug = _slugify_name(run_label)
    run_root = os.path.join(script_dir, "coverage_results", f"{timestamp}_{run_slug}_{scenario_slug}")

    layout = {
        "run_root": run_root,
        "coverage_output": os.path.join(run_root, "01_coverage_outputs"),
        "metadata": os.path.join(run_root, "02_run_metadata"),
    }

    for folder_path in layout.values():
        os.makedirs(folder_path, exist_ok=True)

    return layout

def make_digital_twin(scenario_dir, polygon_gdf):

        buildings_path = os.path.join(scenario_dir, "buildings.csv")
        trees_path = os.path.join(scenario_dir, "trees.csv")
        if os.path.exists(buildings_path) and os.path.exists(trees_path):
            regenerate = _ask("Twin exists, regenerate? (y/n)", default="n")
            if regenerate.lower().startswith("n"):
                return
                    
        wkt_str = polygon_gdf.iloc[0].geometry.wkt

        # save AOI for prop
        with open(os.path.join(scenario_dir, "aoi.wkt"), "w") as file:
            file.write(f"{wkt_str}\n")

        # fetch precursor data
        fetch_geodata(wkt_str, scenario_dir)

        # call gprop to generate the digital twin
        os.system(
            f"""'{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'preproc')}' \
                lpc.laz buildings.wkt ground.tif"""
        )

        # move generated files into the scenario folder
        os.rename("buildings.csv", buildings_path)
        os.rename("trees.csv", trees_path)

def run_coverage_analysis_with_outputs(
        script_dir,
        scenario_path,
        all_antennas,
        polygons_gdf,
        coverage_points,
        path_loss_params,
        run_label,
        point_heights,
        antenna_heights,
        diffraction_losses,
):
    """
    Run coverage analysis and store outputs in a dedicated, timestamped run folder.
    """
    from coverage_app.cli_utils import _info, _ok, _warn

    layout = _build_coverage_run_layout(script_dir, scenario_path, run_label)
    run_root = layout["run_root"]
    coverage_output = layout["coverage_output"]
    metadata_folder = layout["metadata"]

    scenario_copy = os.path.join(metadata_folder, "scenario_used.geojson")
    if scenario_path and os.path.isfile(scenario_path):
            try:
                shutil.copy2(scenario_path, scenario_copy)
            except Exception as exc:
                _warn(f"Could not copy scenario snapshot: {exc}")

    antenna_csv = os.path.join(metadata_folder, "antenna_configuration_used.csv")
    pd.DataFrame(all_antennas).to_csv(antenna_csv, index=False)

    params_json = os.path.join(metadata_folder, "path_loss_parameters.json")
    with open(params_json, "w", encoding="utf-8") as handle:
        json.dump(path_loss_params, handle, indent=2)

    perform_coverage_analysis(
        all_antennas,
        polygons_gdf,
        coverage_points,
        path_loss_params,
        coverage_output,
        point_heights,
        antenna_heights,
        diffraction_losses
    )

    # Legacy mirror for compatibility with existing web/API tools.
    legacy_folder = os.path.join(script_dir, "interactive_results")
    if os.path.exists(legacy_folder):
        shutil.rmtree(legacy_folder, ignore_errors=True)
        shutil.copytree(coverage_output, legacy_folder)

    run_overview = os.path.join(run_root, "RUN_CONTENTS.txt")
    with open(run_overview, "w", encoding="utf-8") as handle:
        handle.write("=== Coverage Run Contents ===\n")
        handle.write(f"Run label: {run_label}\n")
        handle.write(f"Scenario source: {scenario_path}\n\n")
        handle.write("Folders:\n")
        handle.write(f"- Coverage outputs: {coverage_output}\n")
        handle.write(f"- Run metadata: {metadata_folder}\n")
        handle.write(f"- Legacy mirror: {legacy_folder}\n\n")
        handle.write("Key files:\n")
        handle.write(f"- Combined best-data CSV: {os.path.join(coverage_output, 'best_data.csv')}\n")
        handle.write(f"- Scenario snapshot: {scenario_copy}\n")
        handle.write(f"- Antenna configuration: {antenna_csv}\n")
        handle.write(f"- Path loss parameters: {params_json}\n")

    _ok(f"Coverage run folder: {run_root}")
    _info(f"Coverage outputs: {coverage_output}")
    _info(f"Run metadata: {metadata_folder}")
    _info(f"Legacy mirror updated: {legacy_folder}")
    return run_root

def _wait_for_web_save(prompt, expected_path, browser_url):
    """
        Wait for the user to press ENTER, then verify that *expected_path* was
        created or modified by the web app.  If the file was not saved, warn and
        offer to re-open the browser (the Flask server is still running at that
        point) before trying again.  Returns once the file is confirmed saved, or
        when the user explicitly chooses to skip.
        """
    mtime_before = os.path.getmtime(expected_path) if os.path.isfile(expected_path) else None
    while True:
        _wait(prompt)
        saved = False
        if os.path.isfile(expected_path):
            mtime_after = os.path.getmtime(expected_path)
            saved = (mtime_before is None or mtime_after > mtime_before)
        if saved:
                return
        _warn("The scenario file was not saved in the web app.")
        choice = _ask(
            "  [r] Return to the browser and save   [s] Skip (continue with original file)",
            default="r",
        ).strip().lower()
        if choice.startswith("s"):
            return
        webbrowser.open(browser_url)


def add_virtual_antennas(scenario_path, scenario_dir):
    skip = _ask(
        "Analyze the scenario as-is without virtual antennas? [y/n]",
        default="n"
    ).lower().startswith("y")
    
    add_random = False

    # check if the user wants to add virtual antennas
    if not skip:
        method = _ask(
            "Randomly generate virtual antennas (r) or use web customizer (c)?",
            default="r"
        ).lower()
        
        # manually place virtual antennas
        if method.startswith("c"):
            # open current scenario with web viewer, save to new file
            output = os.path.join(scenario_dir, "virtual_scenario.geojson")
            configure_scenario_files(scenario_path, output)
            flask_thread = threading.Thread(target=run_flask_app, daemon=True)
            flask_thread.start()
            time.sleep(2)
            webbrowser.open("http://127.0.0.1:5000/custom_scenario")
            _wait_for_web_save(
                "Add virtual antennas in the web app, save, then press ENTER.",
                output,
                "http://127.0.0.1:5000/custom_scenario",
            )
            _shutdown_flask()
            scenario_path = output
        elif method.startswith("r"): add_random = True

    # load whatever scenario file was decided on, and split antennas
    polygons_gdf, points_gdf = load_geojson_as_gdf(scenario_path)
    real_antennas, virtual_antennas = combine_scenario_antennas(points_gdf)

    # add random virtual antennas if necessary
    if add_random:
        params = _collect_vstation_params(polygons_gdf, real_antennas)
        real_freqs = [ant["frequency"] for ant in real_antennas]
        virtual_antennas.extend(generate_vstations(*params, real_freqs))

    # return AOI, all antennas, real antennas, and virtual antennas
    return [polygons_gdf, real_antennas, virtual_antennas, scenario_path]

def make_gprop_topology(all_antennas):
    # we need lon, lat, height, azimuth, downtilt, frequency, power, gain?
        with open("topology.csv", "w") as file:
            for ant in all_antennas:
                file.write(f"{ant['lon']},")
                file.write(f"{ant['lat']},")
                file.write(f"{ant['height']},")
                file.write(f"{ant['azimuth']},")
                file.write(f"{ant['downtilt']},")
                file.write(f"{ant['frequency']},")
                file.write(f"{ant['Ptx']},")
                file.write(f"{ant['Gtx']}\n")

def make_gprop_points(coverage_points):
    with open("points.csv", "w") as file:
        for curr in coverage_points:
            point = shapely.geometry.Point(curr["lon"], curr["lat"])
            point = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:3857").iloc[0]
            file.write(f"{point.x},{point.y}\n")

def read_diffraction_tbl():
    heights = []
    losses = []
    with open("diff-table.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            heights.append(float(row[0]))
            losses.append(float(row[1]))
    return [heights, losses]

def run_simulation(scenario_path, run_label):
    scenario_dir = os.path.dirname(os.path.abspath(scenario_path))
    script_dir =  os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.isfile(scenario_path):
        _err("scenario.geojson not found. Exiting.")
        return

    polygons_gdf, real_antennas, virtual_antennas, scenario_path = add_virtual_antennas(
        scenario_path,
        scenario_dir
    )

    all_antennas = real_antennas + virtual_antennas
    
    coverage_points = generate_coverage_points(
        polygons_gdf,
        all_antennas,
        default_n=3000
    )
    
    make_digital_twin(scenario_dir, polygons_gdf)

    # generate diffraction table
    buildings_path = os.path.join(scenario_dir, "buildings.csv")
    trees_path = os.path.join(scenario_dir, "trees.csv")
    aoi_path = os.path.join(scenario_dir, "aoi.wkt")
    dsm_path = os.path.join(scenario_dir, "ground.tif")
    make_gprop_topology(all_antennas)
    make_gprop_points(coverage_points)
    os.system(
        f"""'{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prop')}' \
        topology.csv '{aoi_path}' '{buildings_path}' '{trees_path}' '{dsm_path}'"""
    )
    point_heights, diffraction_losses = read_diffraction_tbl()

    antenna_heights = []
    with open("antenna_heights", "r") as file:
        for line in file:
            antenna_heights.append(float(line))

    run_coverage_analysis_with_outputs(
        script_dir=script_dir,
        scenario_path=scenario_path,
        all_antennas=all_antennas,
        polygons_gdf=polygons_gdf,
        coverage_points=coverage_points,
        path_loss_params=choose_path_loss_model(),
        run_label=run_label,
        point_heights=point_heights,
        antenna_heights=antenna_heights,
        diffraction_losses=diffraction_losses
    )

def main():
    global BUFFALO_MODE

    script_dir = os.path.dirname(os.path.abspath(__file__))

    _banner()
    print(f"  {BOLD}Select an option:{RST}\n")
    print(f"    {CYAN}1{RST}  Create a new scenario")
    print(f"    {CYAN}2{RST}  Load an existing .geojson scenario")
    print(f"    {CYAN}3{RST}  Buffalo example scenario (Fruit Belt)")
    print(f"    {CYAN}4{RST}  Optimize channel assignment")
    print()
    choice = _ask("Choice [1-4]").strip()

    # ---------------------------------------------------------------
    # OPTION 1: CREATE NEW SCENARIO
    # ---------------------------------------------------------------
    if choice == "1":
        _section("Create New Scenario")
        _info("Starting Flask server and opening the scenario creator…")

        scenario_geojson_path = os.path.join(script_dir, "scenario.geojson")
        # Remove any stale file so _wait_for_web_save can detect a fresh save
        if os.path.isfile(scenario_geojson_path):
            os.remove(scenario_geojson_path)
            # Tell Flask where save_geojson() should write to
            configure_scenario_files(current_scenario_file=None, output_scenario_file=scenario_geojson_path)

        flask_thread = threading.Thread(target=run_flask_app)
        flask_thread.daemon = True
        flask_thread.start()
        time.sleep(2)
        webbrowser.open("http://127.0.0.1:5000/")
        _wait_for_web_save(
            "Draw your area and antennas, save the scenario, then press ENTER to continue.",
            scenario_geojson_path,
            "http://127.0.0.1:5000/",
        )

        _info("Shutting down Flask server…")
        _shutdown_flask()

        run_simulation(scenario_geojson_path, "create_new_scenario")


    # ---------------------------------------------------------------
    # OPTION 2: LOAD EXISTING SCENARIO
    # ---------------------------------------------------------------
    elif choice == "2":
        _section("Load Existing Scenario")
        existing_files, search_dir = list_geojson_files()
        if not existing_files:
            _err("No .geojson files found.")
            _info(f"Expected location:  {search_dir}")
            _info("Place your .geojson file there and re-run.")
            _info("(Tip: use option 1 to create a new scenario with the web editor — it saves there automatically.)")
            return

        if len(existing_files) == 1:
            chosen_file = existing_files[0]
            _ok(f"Found: {os.path.basename(chosen_file)}")
        else:
            _info("Multiple .geojson files found:")
            for i, f in enumerate(existing_files, start=1):
                print(f"    {CYAN}{i}{RST}  {os.path.basename(f)}")
                sel = _ask("Choose file by number")
            try:
                chosen_file = existing_files[int(sel) - 1]
            except (ValueError, IndexError):
                _err("Invalid choice. Exiting.")
                return
            _ok(f"Using: {os.path.basename(chosen_file)}")

        run_simulation(chosen_file, "load_existing_scenario")


    # ---------------------------------------------------------------
    # OPTION 3: BUFFALO EXAMPLE
    # ---------------------------------------------------------------
    elif choice == "3":
        BUFFALO_MODE = True
        _section("Buffalo Fruit Belt Scenario")
        buffalo_dir = os.path.join(script_dir, "buffalo_scenario")
        fruit_belt_path = os.path.join(buffalo_dir, "fruit_belt.geojson")

        if not os.path.isfile(fruit_belt_path):
            _err("fruit_belt.geojson not found in buffalo_scenario/. Exiting.")
            return
        _ok("Loaded Fruit Belt (Buffalo, NY)")

        run_simulation(fruit_belt_path, "buffalo_fruit_belt")


    # ---------------------------------------------------------------
    # OPTION 4: OPTIMIZE A SCENARIO
    # ---------------------------------------------------------------
    elif choice == "4":
        _section("Channel Assignment Optimization")
        optimize_scenario()

    else:
        _err(f"'{choice}' is not a valid option.")

if __name__ == "__main__":
    main()

