# Modal Render API

The Modal endpoint accepts POST requests and returns a rendered
simulation video with aerodynamic coefficients.

## Request

POST to the endpoint URL (set via `MODAL_RENDER_ENDPOINT` env var).

```json
{
  "job_id": "my-run-001",
  "wind_speed": 1.5,
  "viz_mode": 1,
  "collision_mode": 1,
  "duration": 10,
  "model": "ahmed25",
  "reynolds": 0,
  "obj_data": null,
  "callback_url": null
}
```

| Field          | Type   | Default    | Notes                                    |
|----------------|--------|------------|------------------------------------------|
| job_id         | string | auto UUID  | Used as the S3 filename                  |
| wind_speed     | float  | 1.0        | Freestream velocity, 0-5                 |
| viz_mode       | int    | 1          | 0=depth, 1=velocity mag, 2=direction,    |
|                |        |            | 3=lifetime, 4=turbulence, 5=flow, 6=vort |
| collision_mode | int    | 1          | 0=off, 1=AABB, 2=per-triangle           |
| duration       | int    | 10         | Simulation length in seconds             |
| model          | string | "car"      | "car", "ahmed25", "ahmed35", or "custom" |
| reynolds       | float  | 0          | 0 means use default viscosity            |
| obj_data       | string | null       | Base64-encoded OBJ file (for custom)     |
| callback_url   | string | null       | POST results here when done              |

## Response

```json
{
  "status": "complete",
  "video_url": "https://fluid-sim-renders.s3.eu-west-2.amazonaws.com/renders/my-run-001.mp4",
  "model": "ahmed25",
  "wind_speed": 1.5,
  "cd_value": 0.287,
  "cl_value": 0.041,
  "cd_series": [0.31, 0.29, 0.28, ...],
  "cl_series": [0.04, 0.04, 0.04, ...],
  "error": null
}
```

| Field     | Type        | Notes                                     |
|-----------|-------------|-------------------------------------------|
| status    | string      | "complete" or "error"                     |
| video_url | string      | S3 URL to the rendered MP4                |
| cd_value  | float/null  | Average Cd from the last 5 measurements   |
| cl_value  | float/null  | Average Cl from the last 5 measurements   |
| cd_series | float[]     | All Cd values (after 3s warmup)           |
| cl_series | float[]     | All Cl values (after 3s warmup)           |
| error     | string/null | Error message if status="error"           |

## Error cases

- OBJ file too large (>5 MB base64): 413
- Rate limited (1 request per 60s per IP): 429
- Backend not configured: 503
- Simulation crash or timeout: status="error" with details

## Local testing

```bash
modal run simulation/modal_worker.py \
  --wind=1.5 --model=ahmed25 --duration=10
```
