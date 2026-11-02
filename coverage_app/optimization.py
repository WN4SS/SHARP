"""
optimization.py

Methods for scenario optimization through channel assignment.
"""

import itertools
import os
import random
import shutil
import threading
import time
import webbrowser
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd

from .coverage_analysis import combine_scenario_antennas
from .plotting import bandwidth_mhz_of
from .coverage_analysis import generate_coverage_points
from .coverage_analysis import perform_coverage_analysis
from .coverage_analysis import process_metrics
from .flask_routes import run_flask_app, stop_flask_app
from .path_loss import choose_path_loss_model
from .plotting import plot_performance_history

DEFAULT_SERVICE_THRESHOLDS = {
	"sinr": 0.0,
	"rsrp": -95.0,
	"rsrq": -14.0,
}

DEFAULT_CBRS_CHANNELS = tuple(range(3550, 3700, 10))

METRIC_TOLERANCES = {
	"service_ratio": 1e-6,
	"sinr_p10": 1e-3,
	"sinr": 1e-3,
}


def list_geojson_files():
	"""
	Return a list of .geojson files in the Coverage App directory (same
	location as main.py and where geojson_creator saves scenario.geojson).
	"""
	search_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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


def ask_int_in_range(prompt, default, min_value=1, max_value=None):
	from .cli_utils import _ask, _warn

	while True:
		raw = _ask(prompt, default=default)
		if raw == "":
			value = default
		else:
			try:
				value = int(raw)
			except ValueError:
				_warn("Please enter an integer value.")
				continue

		if value < min_value:
			_warn(f"Value must be >= {min_value}.")
			continue
		if max_value is not None and value > max_value:
			_warn(f"Value must be <= {max_value}.")
			continue
		return value


def format_channel(channel):
	channel = float(channel)
	if channel.is_integer():
		return str(int(channel))
	return f"{channel:.3f}".rstrip("0").rstrip(".")


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


def create_optimization_run_layout(script_dir, scenario_path):
	"""
	Create a dedicated output layout for one optimization run.
	"""
	scenario_name = os.path.splitext(os.path.basename(scenario_path))[0] if scenario_path else "scenario"
	scenario_slug = _slugify_name(scenario_name)
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_root = os.path.join(script_dir, "optimization_results", f"{timestamp}_{scenario_slug}")

	layout = {
		"run_root": run_root,
		"baseline": os.path.join(run_root, "01_baseline_coverage"),
		"optimized": os.path.join(run_root, "02_optimized_coverage"),
		"reports": os.path.join(run_root, "03_reports"),
		"plots": os.path.join(run_root, "04_plots"),
		"baseline_temp": os.path.join(run_root, "_temp_baseline_metrics"),
		"temp": os.path.join(run_root, "_temp_channel_evaluation"),
	}

	for folder_path in layout.values():
		os.makedirs(folder_path, exist_ok=True)

	return layout


def choose_available_channels(current_channels):
	from .cli_utils import _ask, _info, _ok, _section, _warn

	_section("Step 3 / 7 — Available CBRS Channels")
	if current_channels:
		formatted = ", ".join(format_channel(freq) for freq in current_channels)
		_info(f"Current scenario channels: {formatted} MHz")
	_info("Default pool uses nominal 10 MHz CBRS channels from 3550 to 3690 MHz.")
	default_text = ", ".join(format_channel(freq) for freq in DEFAULT_CBRS_CHANNELS)

	while True:
		raw = _ask("Available channels in MHz (comma-separated)", default=default_text)
		tokens = [token.strip() for token in (raw or default_text).split(",") if token.strip()]
		parsed_channels = []
		invalid_tokens = []

		for token in tokens:
			try:
				value = float(token)
			except ValueError:
				invalid_tokens.append(token)
				continue
			parsed_channels.append(int(value) if value.is_integer() else value)

		if invalid_tokens:
			_warn(f"Invalid channel value(s): {', '.join(invalid_tokens)}")
			continue

		available_channels = sorted(set(parsed_channels), key=float)
		if not available_channels:
			_warn("Please provide at least one valid channel.")
			continue

		formatted = ", ".join(format_channel(freq) for freq in available_channels)
		_ok(f"Using {len(available_channels)} channel(s): {formatted} MHz")
		return available_channels


def build_antenna_metadata(antennas):
	station_ids = {}
	antenna_counters = {}
	metadata = []

	for ant in antennas:
		station_key = (round(float(ant["lat"]), 8), round(float(ant["lon"]), 8))
		station_id = station_ids.setdefault(station_key, len(station_ids) + 1)
		antenna_counters[station_key] = antenna_counters.get(station_key, 0) + 1
		antenna_id = antenna_counters[station_key]
		metadata.append({
			"station_id": station_id,
			"antenna_id": antenna_id,
			"label": f"BS #{station_id} / Antenna {antenna_id}",
			"lat": ant["lat"],
			"lon": ant["lon"],
			"type": ant.get("type", "unknown"),
		})

	return metadata


def apply_channel_assignment(antennas, assignment):
	assigned_antennas = []
	for ant, channel in zip(antennas, assignment):
		assigned_ant = dict(ant)
		assigned_ant["frequency"] = channel
		assigned_antennas.append(assigned_ant)
	return assigned_antennas


def build_sampled_channel_plans(available_channels, num_antennas, max_plans, excluded_assignments=None):
	plans = []
	seen = set(excluded_assignments or [])

	if not available_channels or num_antennas <= 0 or max_plans <= 0:
		return plans

	seed_candidates = []
	seed_candidates.append(tuple(available_channels[i % len(available_channels)] for i in range(num_antennas)))
	for offset in range(len(available_channels)):
		seed_candidates.append(
			tuple(available_channels[(i + offset) % len(available_channels)] for i in range(num_antennas))
		)
	for channel in available_channels:
		seed_candidates.append(tuple(channel for _ in range(num_antennas)))

	for candidate in seed_candidates:
		if candidate in seen:
			continue
		seen.add(candidate)
		plans.append(candidate)
		if len(plans) >= max_plans:
			return plans

	rng = random.Random(0)
	attempts = 0
	max_attempts = max_plans * 50
	while len(plans) < max_plans and attempts < max_attempts:
		candidate = tuple(rng.choice(available_channels) for _ in range(num_antennas))
		if candidate not in seen:
			seen.add(candidate)
			plans.append(candidate)
		attempts += 1

	return plans


def format_performance_summary(perf):
	return (
		f"Service: {perf['service_ratio'] * 100:.1f}%  "
		f"P10 SINR: {perf['sinr_p10']:.2f} dB  "
		f"Mean SINR: {perf['sinr']:.2f} dB  "
		f"Mean RSRP: {perf['rsrp']:.2f} dBm  "
		f"Mean RSRQ: {perf['rsrq']:.2f} dB  "
		f"Mean throughput: {perf['throughput']:.2f} Mbps"
	)


def compute_plan_performance(antennas, points, output_folder, path_loss_params):
	if os.path.exists(output_folder):
		shutil.rmtree(output_folder, ignore_errors=True)
	os.makedirs(output_folder, exist_ok=True)

	process_metrics(
		antennas=antennas,
		points=points,
		output_folder=output_folder,
		path_loss_model=path_loss_params["model"],
		base_height=path_loss_params["base_height"],
		mobile_height=path_loss_params["mobile_height"],
		alpha=path_loss_params["alpha"],
		beta=path_loss_params["beta"],
		n=path_loss_params["n"],
		shadowing=path_loss_params["shadowing"],
		omni=path_loss_params["omni"],
		environment=path_loss_params["environment"],
		# ACI: pass all antennas (all frequencies) when enabled
		all_antennas_for_aci=(antennas if path_loss_params.get("enable_aci") else None),
		aclr_db=path_loss_params.get("aclr_db", 45.0),
		acs_db=path_loss_params.get("acs_db", 33.0),
		channel_bw_mhz=path_loss_params.get("channel_bw_mhz", 10.0),
		scenario=path_loss_params.get("scenario", "uma"),
	)
	return compute_average_performance(output_folder, service_thresholds=DEFAULT_SERVICE_THRESHOLDS)


def is_better_channel_assignment(new_perf, old_perf):
	if old_perf is None:
		return True

	sinr_delta = new_perf["sinr"] - old_perf["sinr"]
	if sinr_delta > METRIC_TOLERANCES["sinr"]:
		return True
	if sinr_delta < -METRIC_TOLERANCES["sinr"]:
		return False

	service_delta = new_perf["service_ratio"] - old_perf["service_ratio"]
	if service_delta > METRIC_TOLERANCES["service_ratio"]:
		return True
	if service_delta < -METRIC_TOLERANCES["service_ratio"]:
		return False

	p10_delta = new_perf["sinr_p10"] - old_perf["sinr_p10"]
	return p10_delta > METRIC_TOLERANCES["sinr_p10"]


def save_channel_assignment_report(
	output_folder,
	antenna_metadata,
	initial_assignment,
	best_assignment,
	baseline_perf,
	best_perf,
	evaluated_plans,
	search_mode,
):
	rows = []
	for meta, initial_channel, best_channel in zip(antenna_metadata, initial_assignment, best_assignment):
		rows.append({
			"Base Station": meta["station_id"],
			"Antenna": meta["antenna_id"],
			"Label": meta["label"],
			"Type": meta["type"],
			"Latitude": meta["lat"],
			"Longitude": meta["lon"],
			"Initial Channel (MHz)": initial_channel,
			"Best Channel (MHz)": best_channel,
			"Channel Changed": initial_channel != best_channel,
		})

	csv_path = os.path.join(output_folder, "channel_assignment_comparison.csv")
	pd.DataFrame(rows).to_csv(csv_path, index=False)

	txt_path = os.path.join(output_folder, "optimization_summary.txt")
	with open(txt_path, "w", encoding="utf-8") as handle:
		handle.write("=== Channel Assignment Optimization ===\n")
		handle.write(f"Search mode: {search_mode}\n")
		handle.write(f"Evaluated channel plans: {evaluated_plans}\n")
		handle.write(f"Baseline mean SINR: {baseline_perf['sinr']:.2f} dB\n")
		handle.write(f"Best mean SINR: {best_perf['sinr']:.2f} dB\n")
		handle.write(f"Baseline service ratio: {baseline_perf['service_ratio'] * 100:.1f}%\n")
		handle.write(f"Best service ratio: {best_perf['service_ratio'] * 100:.1f}%\n")
		handle.write("\nBest channel plan:\n")
		for meta, best_channel in zip(antenna_metadata, best_assignment):
			handle.write(f"- {meta['label']}: {format_channel(best_channel)} MHz\n")

	return csv_path, txt_path


def optimize_scenario():
	"""
	Search channel assignments for the scenario antennas and keep the plan
	that maximizes mean SINR over the AOI.
	"""
	from .cli_utils import _ask, _err, _info, _ok, _section, _wait, _warn, CYAN, RST

	script_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

	_section("Step 1 / 7 — Create or Load Scenario")
	print(f"    {CYAN}A{RST}  Create a new scenario with the web editor")
	print(f"    {CYAN}B{RST}  Load an existing .geojson scenario")
	print()
	choice = _ask("Choice [A/B]").strip().lower()

	scenario_geojson_path = os.path.join(script_dir, "scenario.geojson")
	final_scenario = None

	if choice == "a":
		flask_thread = threading.Thread(target=run_flask_app)
		flask_thread.start()
		time.sleep(2)
		webbrowser.open("http://127.0.0.1:5000/")
		_wait("Draw your area and antennas, save the scenario, then press ENTER.")
		if stop_flask_app(timeout=5):
			flask_thread.join(timeout=5)
		else:
			_warn("Flask shutdown may not have completed cleanly before optimization continued.")

		if not os.path.isfile(scenario_geojson_path):
			_err("No scenario.geojson was created. Exiting optimization.")
			_info(f"Expected location: {scenario_geojson_path}")
			return
		final_scenario = scenario_geojson_path

	elif choice == "b":
		existing_files, search_dir = list_geojson_files()
		if not existing_files:
			_err("No .geojson files found.")
			_info(f"Expected location:  {search_dir}")
			_info("Place your .geojson file there and re-run.")
			_info("(Tip: use option 1 from the main menu to create one with the web editor.)")
			return
		if len(existing_files) == 1:
			final_scenario = existing_files[0]
			_ok(f"Found: {os.path.basename(final_scenario)}")
		else:
			_info("Available .geojson files:")
			for i, path in enumerate(existing_files, start=1):
				print(f"    {CYAN}{i}{RST}  {os.path.basename(path)}")
			sel = _ask("Choose file by number")
			try:
				final_scenario = existing_files[int(sel) - 1]
			except (ValueError, IndexError):
				_err("Invalid choice. Exiting.")
				return
			_ok(f"Using: {os.path.basename(final_scenario)}")
	else:
		_err("Invalid choice. Aborting optimization.")
		return

	run_layout = create_optimization_run_layout(script_dir, final_scenario)
	run_root = run_layout["run_root"]
	reports_folder = run_layout["reports"]
	_info(f"Optimization run folder: {run_root}")

	scenario_copy_path = os.path.join(reports_folder, "scenario_used_for_optimization.geojson")
	try:
		shutil.copy2(final_scenario, scenario_copy_path)
		_info(f"Scenario snapshot saved at: {scenario_copy_path}")
	except Exception as exc:
		_warn(f"Could not copy scenario snapshot: {exc}")

	_section("Step 2 / 7 — Load Scenario & Coverage Points")
	polygons_gdf, points_gdf = load_geojson_as_gdf(final_scenario)
	real_antennas, virtual_antennas = combine_scenario_antennas(points_gdf)
	all_antennas = real_antennas + virtual_antennas
	if not all_antennas:
		_err("No antennas found in the selected scenario. Optimization aborted.")
		return

	antenna_metadata = build_antenna_metadata(all_antennas)
	station_count = len({meta["station_id"] for meta in antenna_metadata})
	_info(f"Found {len(all_antennas)} antenna(s) across {station_count} base station(s).")

	coverage_points = generate_coverage_points(polygons_gdf, all_antennas, default_n=3000)
	initial_assignment = tuple(ant["frequency"] for ant in all_antennas)
	current_channels = sorted(set(initial_assignment), key=float)

	available_channels = choose_available_channels(current_channels)
	total_assignments = len(available_channels) ** len(all_antennas)
	baseline_in_pool = all(freq in available_channels for freq in initial_assignment)
	_info(
		f"{len(all_antennas)} antenna(s) × {len(available_channels)} available channel(s) "
		f"→ {total_assignments:,} possible channel plans."
	)

	default_iterations = min(100, total_assignments)
	max_iterations = ask_int_in_range(
		f"Max channel plans to test (total plans: {total_assignments:,})",
		default=default_iterations,
		min_value=1,
		max_value=total_assignments,
	)

	_section("Step 4 / 7 — Initial Performance Baseline")
	path_loss_params = choose_path_loss_model()

	initial_folder = run_layout["baseline"]
	baseline_metrics_folder = run_layout["baseline_temp"]
	init_perf = compute_plan_performance(
		antennas=all_antennas,
		points=coverage_points,
		output_folder=baseline_metrics_folder,
		path_loss_params=path_loss_params,
	)
	_ok(f"Initial metrics  -> {format_performance_summary(init_perf)}")

	_section("Step 5 / 7 — Channel-Assignment Search")
	if total_assignments <= max_iterations:
		candidate_count = total_assignments - (1 if baseline_in_pool else 0)
		channel_plans = (
			assignment
			for assignment in itertools.product(available_channels, repeat=len(all_antennas))
			if not (baseline_in_pool and assignment == initial_assignment)
		)
		search_mode = "exhaustive"
		_info("Evaluating every valid channel assignment in the provided pool.")
	else:
		channel_plans = build_sampled_channel_plans(
			available_channels,
			len(all_antennas),
			max_iterations,
			excluded_assignments={initial_assignment} if baseline_in_pool else None,
		)
		candidate_count = len(channel_plans)
		search_mode = "sampled"
		_warn(
			f"Sampling {candidate_count} channel plans out of {total_assignments:,} total plans "
			"to keep the search tractable…"
		)

	_info(f"Channel plans to test: {candidate_count}")
	if not baseline_in_pool:
		_warn("The current scenario uses channels outside the available pool; baseline is reference-only.")

	best_perf = init_perf if baseline_in_pool else None
	best_assignment = initial_assignment if baseline_in_pool else None
	best_antennas = [dict(ant) for ant in all_antennas] if baseline_in_pool else None
	best_plan_tag = "initial" if baseline_in_pool else None

	performance_history = [{
		"iteration": 0,
		"tag": "initial",
		"SINR": init_perf["sinr"],
		"RSRP": init_perf["rsrp"],
		"RSRQ": init_perf["rsrq"],
		"Throughput": init_perf["throughput"],
		"ServiceRatio": init_perf["service_ratio"],
		"SINR_P10": init_perf["sinr_p10"],
	}]

	temp_eval_folder = run_layout["temp"]
	start_time = time.time()
	evaluated_plans = 0

	if candidate_count == 0:
		_info("No alternative channel plans remain after excluding the current assignment.")

	for idx, assignment in enumerate(channel_plans, start=1):
		current_tag = f"channel_plan_{idx}"
		print(f"  {CYAN}▶{RST}  Testing channel plan {idx}/{candidate_count}…")

		candidate_antennas = apply_channel_assignment(all_antennas, assignment)
		curr_perf = compute_plan_performance(
			antennas=candidate_antennas,
			points=coverage_points,
			output_folder=temp_eval_folder,
			path_loss_params=path_loss_params,
		)
		print(
			f"Channel plan {idx} => Mean SINR: {curr_perf['sinr']:.2f} dB, "
			f"Service: {curr_perf['service_ratio'] * 100:.1f}%, "
			f"P10 SINR: {curr_perf['sinr_p10']:.2f} dB"
		)

		if is_better_channel_assignment(curr_perf, best_perf):
			_ok(f"Channel plan {idx} is better — updating best.")
			best_perf = curr_perf
			best_assignment = assignment
			best_antennas = candidate_antennas
			best_plan_tag = current_tag

		performance_history.append({
			"iteration": idx,
			"tag": current_tag,
			"SINR": curr_perf["sinr"],
			"RSRP": curr_perf["rsrp"],
			"RSRQ": curr_perf["rsrq"],
			"Throughput": curr_perf["throughput"],
			"ServiceRatio": curr_perf["service_ratio"],
			"SINR_P10": curr_perf["sinr_p10"],
		})
		evaluated_plans += 1

	if os.path.exists(temp_eval_folder):
		shutil.rmtree(temp_eval_folder, ignore_errors=True)
	if os.path.exists(baseline_metrics_folder):
		shutil.rmtree(baseline_metrics_folder, ignore_errors=True)

	if best_assignment is None or best_antennas is None or best_perf is None:
		_err("No valid channel assignment was evaluated. Optimization aborted.")
		return

	end_time = time.time()
	elapsed_time = end_time - start_time
	hours = int(elapsed_time // 3600)
	minutes = int((elapsed_time % 3600) // 60)
	seconds = elapsed_time % 60

	_section("Step 6 / 7 — Best Channel Assignment")
	_ok(f"Best plan: {best_plan_tag}")
	_ok(f"Best metrics  -> {format_performance_summary(best_perf)}")

	changed_channels = 0
	for meta, initial_channel, best_channel in zip(antenna_metadata, initial_assignment, best_assignment):
		if initial_channel == best_channel:
			continue
		changed_channels += 1
		print(
			f"    {CYAN}{meta['label']}{RST}  "
			f"{format_channel(initial_channel)} → {format_channel(best_channel)} MHz"
		)
	if changed_channels == 0:
		_info("No channel changes were required for the best evaluated plan.")

	_section("Step 7 / 7 — Final Plots")
	perform_coverage_analysis(
		all_antennas,
		polygons_gdf,
		coverage_points,
		path_loss_params,
		output_folder=initial_folder,
		skip_grid_prompt=True,
		grid_granularity=50,
		title_suffix="_initial",
	)

	optimal_folder = run_layout["optimized"]
	os.makedirs(optimal_folder, exist_ok=True)

	if best_assignment == initial_assignment:
		_info("The current channel plan was already optimal within the evaluated search space.")
		perform_coverage_analysis(
			all_antennas,
			polygons_gdf,
			coverage_points,
			path_loss_params,
			output_folder=optimal_folder,
			skip_grid_prompt=True,
			grid_granularity=50,
			title_suffix="_optimal",
		)
	else:
		_info("Generating plots for the best channel assignment…")
		perform_coverage_analysis(
			best_antennas,
			polygons_gdf,
			coverage_points,
			path_loss_params,
			output_folder=optimal_folder,
			skip_grid_prompt=True,
			grid_granularity=50,
			title_suffix="_optimal",
		)

	report_csv, report_txt = save_channel_assignment_report(
		output_folder=reports_folder,
		antenna_metadata=antenna_metadata,
		initial_assignment=initial_assignment,
		best_assignment=best_assignment,
		baseline_perf=init_perf,
		best_perf=best_perf,
		evaluated_plans=evaluated_plans,
		search_mode=search_mode,
	)

	history_plot_path = plot_performance_history(performance_history, run_layout["plots"])

	run_overview_path = os.path.join(run_root, "RUN_CONTENTS.txt")
	with open(run_overview_path, "w", encoding="utf-8") as handle:
		handle.write("=== Optimization Run Contents ===\n")
		handle.write(f"Scenario source: {final_scenario}\n")
		handle.write(f"Scenario snapshot: {scenario_copy_path}\n\n")
		handle.write("Folders:\n")
		handle.write(f"- 01_baseline_coverage: {initial_folder}\n")
		handle.write(f"- 02_optimized_coverage: {optimal_folder}\n")
		handle.write(f"- 03_reports: {reports_folder}\n")
		handle.write(f"- 04_plots: {run_layout['plots']}\n\n")
		handle.write("Key files:\n")
		handle.write(f"- Channel assignment CSV: {report_csv}\n")
		handle.write(f"- Optimization summary: {report_txt}\n")
		handle.write(f"- Performance history plot: {history_plot_path}\n")

	_section("Optimization Complete")
	_ok(f"Run root folder       → {run_root}")
	_ok(f"Baseline coverage     → {initial_folder}")
	_ok(f"Optimized coverage    → {optimal_folder}")
	_ok(f"Reports folder        → {reports_folder}")
	_ok(f"Performance plots     → {run_layout['plots']}")
	_info(f"Run overview file: {run_overview_path}")
	_info(f"Assignment report: {report_csv}")
	_info(f"Summary report: {report_txt}")
	_info(f"History plot: {history_plot_path}")
	_info(f"Total time: {hours}h {minutes}m {seconds:.2f}s")


def compute_average_performance(folder, service_thresholds=None):
	"""
	Read best_data.csv and compute summary metrics used for optimization.

	Throughput is estimated point-wise as B * log2(1 + SINR_linear), with B the
	channel width the point was served on, then averaged across points. Service
	ratio tracks the share of points meeting the configured SINR/RSRP/RSRQ
	thresholds.
	"""
	if service_thresholds is None:
		service_thresholds = DEFAULT_SERVICE_THRESHOLDS

	csv_path = os.path.join(folder, "best_data.csv")
	if not os.path.isfile(csv_path):
		return {
			"sinr": -999.0,
			"rsrp": -999.0,
			"rsrq": -999.0,
			"throughput": 0.0,
			"service_ratio": 0.0,
			"sinr_p10": -999.0,
		}

	df = pd.read_csv(csv_path)
	if df.empty:
		return {
			"sinr": -999.0,
			"rsrp": -999.0,
			"rsrq": -999.0,
			"throughput": 0.0,
			"service_ratio": 0.0,
			"sinr_p10": -999.0,
		}

	sinr_arr = df["Best Antenna SINR (dB)"].to_numpy(dtype=float)
	bw_arr = bandwidth_mhz_of(df)
	rsrp_arr = df["Best Antenna RSRP (dBm)"].to_numpy(dtype=float)
	rsrq_arr = df["Best Antenna RSRQ (dB)"].to_numpy(dtype=float)
	valid_mask = np.isfinite(sinr_arr) & np.isfinite(rsrp_arr) & np.isfinite(rsrq_arr)
	if not np.any(valid_mask):
		return {
			"sinr": -999.0,
			"rsrp": -999.0,
			"rsrq": -999.0,
			"throughput": 0.0,
			"service_ratio": 0.0,
			"sinr_p10": -999.0,
		}

	sinr_arr = sinr_arr[valid_mask]
	bw_arr = bw_arr[valid_mask]
	rsrp_arr = rsrp_arr[valid_mask]
	rsrq_arr = rsrq_arr[valid_mask]

	avg_sinr = float(np.mean(sinr_arr))
	avg_rsrp = float(np.mean(rsrp_arr))
	avg_rsrq = float(np.mean(rsrq_arr))
	sinr_p10 = float(np.percentile(sinr_arr, 10))
	sinr_lin = np.power(10.0, sinr_arr / 10.0)
	throughput_samples = bw_arr * np.log2(1.0 + np.maximum(sinr_lin, 0.0))
	throughput_mbps = float(np.mean(throughput_samples))
	service_mask = (
		(sinr_arr >= service_thresholds["sinr"])
		& (rsrp_arr >= service_thresholds["rsrp"])
		& (rsrq_arr >= service_thresholds["rsrq"])
	)
	service_ratio = float(np.mean(service_mask))

	return {
		"sinr": avg_sinr,
		"rsrp": avg_rsrp,
		"rsrq": avg_rsrq,
		"throughput": throughput_mbps,
		"service_ratio": service_ratio,
		"sinr_p10": sinr_p10,
	}

