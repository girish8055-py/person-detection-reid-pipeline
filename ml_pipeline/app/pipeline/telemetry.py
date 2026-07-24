"""
pipeline/telemetry.py

Reads a drone's offline_teledata_<flight_id>.csv (GPS/telemetry log) and
answers: "where was the drone, and at what real-world time, when the video
was T seconds in?"

The CSV mixes metadata key/value lines (DroneId, FlightId, planBoundary,
GpsInfo, etc.) in with the actual per-second telemetry rows -- there's no
separate metadata section, so this reader keeps only rows that parse as
real telemetry (same column count as the header, numeric timestamp) and
throws away everything else.
"""

import csv
import os
from datetime import datetime, timezone
from typing import Optional


class TelemetryReader:

    def __init__(self, teledata_csv_path: Optional[str] = None, telemetry_data: Optional[list] = None, video_start_epoch: Optional[float] = None):
        """
        Args:
            teledata_csv_path: path to offline_teledata_<flight_id>.csv
            telemetry_data: list of telemetry dictionaries or list of lists
            video_start_epoch: unix timestamp marking when the VIDEO started
                recording. If not given, defaults to the telemetry log's own
                first timestamp -- i.e. assumes the video started recording
                at the same moment the drone armed. If your video and flight
                log are NOT synced to the same start moment, pass the real
                video start time explicitly.
        """
        self.samples = []  # sorted list of (timestamp, lat, lon, alt, yaw, obs_dist, live_obs_dist)
        if telemetry_data is not None:
            self._parse_list(telemetry_data)
        elif teledata_csv_path is not None:
            self._parse(teledata_csv_path)

        if video_start_epoch is not None:
            self.video_start_epoch = video_start_epoch
        elif self.samples:
            self.video_start_epoch = self.samples[0][0]
        else:
            self.video_start_epoch = None

    def _parse_list(self, data: list) -> None:
        if not data:
            return

        # Check if it is a list of dicts (JSON array of objects) or list of lists (CSV-like)
        first_item = data[0]
        if isinstance(first_item, dict):
            for row in data:
                ts_val = row.get("timestamp")
                if not ts_val:
                    continue
                try:
                    if isinstance(ts_val, (int, float)):
                        ts = float(ts_val)
                    else:
                        ts = datetime.fromisoformat(str(ts_val)).timestamp()
                except ValueError:
                    continue

                item_data = row.get("data") or row
                try:
                    lat = float(item_data.get("lat", 0.0))
                    lon = float(item_data.get("lon", 0.0))
                    alt = float(item_data.get("alt", 0.0))
                    yaw = float(item_data.get("yaw") if item_data.get("yaw") is not None else item_data.get("dir", 0.0))
                    obs_dist = float(item_data.get("ObsDist") if item_data.get("ObsDist") is not None else item_data.get("obsDist", 0.0))
                    live_obs_dist = float(item_data.get("liveObstacleDistance", 0.0))
                except (ValueError, TypeError):
                    continue

                self.samples.append((ts, lat, lon, alt, yaw, obs_dist, live_obs_dist))
        elif isinstance(first_item, list):
            header = [h.strip() for h in first_item]
            try:
                lat_idx = header.index("lat")
                lon_idx = header.index("lon")
                alt_idx = header.index("alt")
                ts_idx = header.index("timestamp")
                yaw_idx = header.index("yaw") if "yaw" in header else (header.index("dir") if "dir" in header else None)
                obs_dist_idx = header.index("obsDist") if "obsDist" in header else None
                live_obs_dist_idx = header.index("liveObstacleDistance") if "liveObstacleDistance" in header else None
            except ValueError:
                return

            for row in data[1:]:
                try:
                    ts_val = row[ts_idx]
                    try:
                        ts = float(ts_val)
                    except ValueError:
                        ts = datetime.fromisoformat(str(ts_val)).timestamp()

                    lat = float(row[lat_idx])
                    lon = float(row[lon_idx])
                    alt = float(row[alt_idx])
                    yaw = float(row[yaw_idx]) if yaw_idx is not None else 0.0
                    obs_dist = float(row[obs_dist_idx]) if obs_dist_idx is not None else 0.0
                    live_obs_dist = float(row[live_obs_dist_idx]) if live_obs_dist_idx is not None else 0.0
                except (ValueError, IndexError, TypeError):
                    continue

                self.samples.append((ts, lat, lon, alt, yaw, obs_dist, live_obs_dist))

        self.samples.sort(key=lambda s: s[0])

    def _parse(self, path: str) -> None:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader)]

            try:
                lat_idx = header.index("lat")
                lon_idx = header.index("lon")
                alt_idx = header.index("alt")
                ts_idx = header.index("timestamp")
                yaw_idx = header.index("yaw") if "yaw" in header else (header.index("dir") if "dir" in header else None)
                obs_dist_idx = header.index("obsDist") if "obsDist" in header else None
                live_obs_dist_idx = header.index("liveObstacleDistance") if "liveObstacleDistance" in header else None
            except ValueError as e:
                raise ValueError(f"Expected telemetry columns not found in {path}: {e}")

            expected_cols = len(header)

            for row in reader:
                if len(row) != expected_cols:
                    continue  # metadata/footer line (DroneId, FlightId, EndTime, ...)
                try:
                    ts = float(row[ts_idx])
                    lat = float(row[lat_idx])
                    lon = float(row[lon_idx])
                    alt = float(row[alt_idx])
                    yaw = float(row[yaw_idx]) if yaw_idx is not None else 0.0
                    obs_dist = float(row[obs_dist_idx]) if obs_dist_idx is not None else 0.0
                    live_obs_dist = float(row[live_obs_dist_idx]) if live_obs_dist_idx is not None else 0.0
                except ValueError:
                    continue  # non-numeric row -- not real telemetry

                self.samples.append((ts, lat, lon, alt, yaw, obs_dist, live_obs_dist))

        self.samples.sort(key=lambda s: s[0])

    def get_position(self, elapsed_seconds: float, distance_meters: float = 10.0) -> Optional[dict]:
        """
        Args:
            elapsed_seconds: seconds since the video started (the same
                `current_time` value your detection pipeline already
                computes per frame).
            distance_meters: fallback distance between drone and person (default 10m).

        Returns:
            dict: {lat, lon, alt, unix_timestamp, iso_time, drone_lat, drone_lon, yaw, distance}, 
            or None if no telemetry was loaded at all.
        """
        if not self.samples or self.video_start_epoch is None:
            return None

        target_ts = self.video_start_epoch + elapsed_seconds
        lat, lon, alt, yaw, obs_dist, live_obs_dist = self._interpolate(target_ts)

        # Decide which distance to use:
        # 1. Use obsDist if it is a valid non-zero distance.
        # 2. Use liveObstacleDistance if it is a valid non-zero distance.
        # 3. Fall back to distance_meters parameter.
        used_distance = distance_meters
        if obs_dist > 0.0:
            used_distance = obs_dist
        elif live_obs_dist > 0.0:
            used_distance = live_obs_dist

        # Calculate person's location based on drone's position, yaw/heading and selected distance
        import math
        R = 6378137.0  # Earth's radius in meters
        bearing = math.radians(yaw)
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)

        lat2 = math.asin(
            math.sin(lat1) * math.cos(used_distance / R) +
            math.cos(lat1) * math.sin(used_distance / R) * math.cos(bearing)
        )
        lon2 = lon1 + math.atan2(
            math.sin(bearing) * math.sin(used_distance / R) * math.cos(lat1),
            math.cos(used_distance / R) - math.sin(lat1) * math.sin(lat2)
        )

        person_lat = math.degrees(lat2)
        person_lon = math.degrees(lon2)

        return {
            "lat": round(person_lat, 7),
            "lon": round(person_lon, 7),
            "alt": round(alt, 2),
            "yaw": round(yaw, 2),
            "distance": round(used_distance, 2),
            "drone_lat": round(lat, 7),
            "drone_lon": round(lon, 7),
            "unix_timestamp": target_ts,
            "iso_time": datetime.fromtimestamp(target_ts, tz=timezone.utc).isoformat(),
        }

    def _interpolate(self, target_ts: float):
        samples = self.samples

        if target_ts <= samples[0][0]:
            return samples[0][1], samples[0][2], samples[0][3], samples[0][4], samples[0][5], samples[0][6]
        if target_ts >= samples[-1][0]:
            return samples[-1][1], samples[-1][2], samples[-1][3], samples[-1][4], samples[-1][5], samples[-1][6]

        for i in range(len(samples) - 1):
            t0 = samples[i][0]
            t1 = samples[i + 1][0]
            if t0 <= target_ts <= t1:
                lat0, lon0, alt0, yaw0, obs0, live0 = samples[i][1], samples[i][2], samples[i][3], samples[i][4], samples[i][5], samples[i][6]
                lat1, lon1, alt1, yaw1, obs1, live1 = samples[i + 1][1], samples[i + 1][2], samples[i + 1][3], samples[i + 1][4], samples[i + 1][5], samples[i + 1][6]
                if t1 == t0:
                    return lat0, lon0, alt0, yaw0, obs0, live0
                frac = (target_ts - t0) / (t1 - t0)
                
                # Interpolating yaw (heading) correctly handling 0-360 wrap-around
                diff = (yaw1 - yaw0 + 180) % 360 - 180
                yaw = (yaw0 + diff * frac) % 360

                obs = obs0 + (obs1 - obs0) * frac
                live = live0 + (live1 - live0) * frac
                
                return (
                    lat0 + (lat1 - lat0) * frac,
                    lon0 + (lon1 - lon0) * frac,
                    alt0 + (alt1 - alt0) * frac,
                    yaw,
                    obs,
                    live
                )

        return samples[-1][1], samples[-1][2], samples[-1][3], samples[-1][4], samples[-1][5], samples[-1][6]


def parse_video_start_epoch(flightdata_path: Optional[str] = None, flightdata_list: Optional[list] = None) -> Optional[float]:
    """
    Parses a flightdata CSV or list of events to find the video start timestamp.
    Looks for messages containing "video" and "visible" (excluding "not visible"),
    or containing "video start" or "start video".
    """
    if flightdata_list is not None:
        try:
            for row in flightdata_list:
                if isinstance(row, dict):
                    message = str(row.get("message") or row.get("data", {}).get("message") or "").strip()
                    ts_val = row.get("timestamp") or row.get("time")
                elif isinstance(row, list) and len(row) >= 3:
                    ts_val, _, message = row[0].strip(), row[1].strip(), row[2].strip()
                else:
                    continue
                
                if not ts_val:
                    continue

                msg_lower = message.lower()
                if "video" in msg_lower:
                    try:
                        if isinstance(ts_val, (int, float)):
                            ts = float(ts_val)
                        else:
                            ts_str = str(ts_val).strip()
                            ts = float(ts_str) if ts_str.replace('.', '', 1).isdigit() else datetime.fromisoformat(ts_str).timestamp()
                    except ValueError:
                        continue

                    if "visible" in msg_lower and "not" not in msg_lower:
                        return ts
                    if "start" in msg_lower or "record" in msg_lower:
                        return ts
        except Exception as e:
            print(f"Error parsing flightdata list for video start: {e}")
        return None

    if not flightdata_path or not os.path.exists(flightdata_path):
        return None
    try:
        with open(flightdata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            # Skip header if present
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    ts_str, msg_type, message = row[0].strip(), row[1].strip(), row[2].strip()
                    msg_lower = message.lower()
                    if "video" in msg_lower:
                        try:
                            ts = float(ts_str) if ts_str.replace('.', '', 1).isdigit() else datetime.fromisoformat(ts_str).timestamp()
                        except ValueError:
                            continue
                        if "visible" in msg_lower and "not" not in msg_lower:
                            return ts
                        if "start" in msg_lower or "record" in msg_lower:
                            return ts
    except Exception as e:
        print(f"Error parsing flightdata CSV for video start: {e}")
    return None