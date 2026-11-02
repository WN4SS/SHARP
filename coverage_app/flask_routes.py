"""
flask_routes.py

Definitions for Flask routes.
"""

import os
import json
import threading


import pandas as pd

from flask import Flask, render_template, request, jsonify, make_response


shutdown_event = threading.Event()
flask_server = None
CURRENT_SCENARIO_FILE = None
OUTPUT_SCENARIO_FILE = None
_state_lock = threading.Lock()

app = Flask(__name__, template_folder='web_templates')
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(APP_DIR, os.pardir))


def configure_scenario_files(current_scenario_file=None, output_scenario_file=None):
	global CURRENT_SCENARIO_FILE, OUTPUT_SCENARIO_FILE
	with _state_lock:
		CURRENT_SCENARIO_FILE = current_scenario_file
		OUTPUT_SCENARIO_FILE = output_scenario_file


def get_scenario_files():
	with _state_lock:
		return CURRENT_SCENARIO_FILE, OUTPUT_SCENARIO_FILE


def _shutdown_werkzeug_server(server):
	global flask_server
	try:
		server.shutdown()
		print("Flask server shutdown completed")
	finally:
		shutdown_event.set()
		if flask_server is server:
			flask_server = None


def _trigger_flask_shutdown():
	server = flask_server
	if server is None:
		shutdown_event.set()
		return False

	shutdown_event.clear()
	threading.Thread(target=_shutdown_werkzeug_server, args=(server,), daemon=True).start()
	return True


def stop_flask_app(timeout=5.0):
	if not _trigger_flask_shutdown():
		return True
	return shutdown_event.wait(timeout=timeout)

@app.route("/")
def geojson_creator():
	return render_template("geojson_creator.html")

@app.route("/custom_scenario")
def custom_scenario():
	return render_template("scenario_customizer.html")

@app.route("/get_current_scenario")
def get_current_scenario():
	scenario_path, _ = get_scenario_files()
	if not scenario_path:
		resp = make_response(jsonify({"features": []}))
		resp.headers['Cache-Control'] = 'no-store'
		return resp

	if not os.path.isabs(scenario_path):
		scenario_path = os.path.join(BASE_DIR, scenario_path)
	if not os.path.isfile(scenario_path):
		resp = make_response(jsonify({"features": []}))
		resp.headers['Cache-Control'] = 'no-store'
		return resp
	with open(scenario_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	resp = make_response(jsonify(data))
	resp.headers['Cache-Control'] = 'no-store'
	return resp

@app.route("/save_geojson", methods=["POST"])
def save_geojson():
	"""
	Saves the newly created real scenario from the web-based editor.
	"""
	try:
		print("Received POST request to /save_geojson")
		data = request.get_json()
		if not data:
			return jsonify({"status": "error", "message": "No GeoJSON data received"}), 400

		if "features" in data:
			for feature in data["features"]:
				if feature.get("geometry", {}).get("type") == "Point":
					if "properties" not in feature:
						feature["properties"] = {}
					if "type" not in feature["properties"]:
						feature["properties"]["type"] = "real_antenna"

		_, output_path = get_scenario_files()
		filename = output_path if output_path else os.path.join(BASE_DIR, "scenario.geojson")
		if not os.path.isabs(filename):
			filename = os.path.join(BASE_DIR, filename)
		with open(filename, "w", encoding="utf-8") as f:
			json.dump(data, f, ensure_ascii=False, indent=2)

		print(f"Real scenario saved successfully at: {filename}")
		return jsonify({"status": "ok", "message": "Real antennas scenario saved successfully"})
	except Exception as e:
		print(f"Error saving GeoJSON: {str(e)}")
		return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/save_custom_scenario", methods=["POST"])
def save_custom_scenario():
	"""
	Saves the scenario from scenario_customizer (with virtual antennas).
	"""
	try:
		data = request.get_json()
		if not data:
			return jsonify({"status": "error", "message": "No scenario data received"}), 400

		current_scenario_path, output_scenario_path = get_scenario_files()
		if not output_scenario_path:
			output_scenario_path = os.path.join(BASE_DIR, "virtual_scenario.geojson")
		elif not os.path.isabs(output_scenario_path):
			output_scenario_path = os.path.join(BASE_DIR, output_scenario_path)

		configure_scenario_files(
			current_scenario_file=current_scenario_path,
			output_scenario_file=output_scenario_path,
		)

		with open(output_scenario_path, "w", encoding="utf-8") as f:
			json.dump(data, f, ensure_ascii=False, indent=2)

		print(f"Virtual scenario saved at: {output_scenario_path}")
		return jsonify({"status": "ok", "message": "Virtual scenario saved."})
	except Exception as e:
		print(f"Error saving virtual scenario: {str(e)}")
		return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/shutdown", methods=["POST"])
def shutdown():
	if _trigger_flask_shutdown():
		return "Server shutting down"
	return "Server not running"


@app.route("/coverage_data")
def coverage_data():
	"""
	API route to fetch coverage data (best coverage per point) in JSON form.
	"""
	output_folder = os.path.join(BASE_DIR, "interactive_results")
	best_data_csv = os.path.join(output_folder, "best_data.csv")
	if not os.path.isfile(best_data_csv):
		return jsonify({"error": "No coverage data found. Please run coverage computation first."}), 404

	df = pd.read_csv(best_data_csv)
	coverage_list = []
	for _, row in df.iterrows():
		coverage_list.append({
			"lat": row["Latitude"],
			"lon": row["Longitude"],
			"sinr_db": row["Best Antenna SINR (dB)"],
			"rsrp_dbm": row["Best Antenna RSRP (dBm)"]
		})
	return jsonify({"coverage_data": coverage_list}), 200


def run_flask_app():
	global flask_server

	from werkzeug.serving import make_server
	shutdown_event.clear()
	flask_server = make_server('127.0.0.1', 5000, app)

	print("Starting Flask server...")
	try:
		flask_server.serve_forever()
	finally:
		flask_server = None
		shutdown_event.set()

