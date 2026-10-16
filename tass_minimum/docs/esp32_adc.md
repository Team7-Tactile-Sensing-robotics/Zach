# ESP32 ADC observation and calibration

## What is known from this repository

No Arduino/ESP32 sketch or real raw-ADC capture was present when this path was
added. Consequently the firmware pin, ADC resolution, attenuation, effective
sample rate, acquisition timing, and analog front-end are **unknown**. The
simulation protocol currently uses 48 kHz, but that is a simulation setting and
must not be presented as the ESP32 rate. ESP32 attenuation is metadata here; it
does not imply a `3.3/4095` conversion because ADC transfer curves, attenuation,
chip variation, loading, and external conditioning can all matter.

The observed simulation signal path is:

```text
string displacement/velocity
  -> simulated piezo gain and voltage noise
  -> piezo_sensor_N_V diagnostic [V]
  -> optional analog gain / high-pass / low-pass / bias / noise
  -> calibrated affine voltage-to-count mapping
  -> code noise / rounding / clipping
  -> piezo_sensor_N_raw_adc [integer count]
```

The no-contact reference follows the same observation path and is saved as
`piezo_sensor_N_reference_V` and `piezo_sensor_N_reference_raw_adc`.

## ADC configuration

Start from `config/esp32_adc_unconfigured.json`. Generation refuses to enable
the observation model without an explicit resolution, positive voltage-to-code
gain, code at zero input voltage, and provenance.

| Field | Unit / role | Status without firmware or calibration |
| --- | --- | --- |
| `resolution_bits` | bits; sets valid codes `0..2^bits-1` | unknown |
| `attenuation_db` | firmware setting recorded as metadata | unknown; never used as an automatic volts conversion |
| `voltage_to_code_gain_counts_per_V` | affine slope at the actual ADC input | unknown; calibrate or configure explicitly |
| `zero_voltage_code` | affine intercept in counts | unknown; calibrate or configure explicitly |
| `analog_gain_V_per_V`, `analog_offset_V` | external conditioning before ADC | configured/assumed until measured |
| voltage/code noise | Gaussian observation-model terms | configured/assumed until measured |
| high/low-pass cutoffs | optional first-order conditioning response, Hz | configured/assumed until measured |
| `parameter_source` | `firmware`, `measured`, `calibrated`, or `assumed` | required when enabled |

The affine mapping is an approximation even after calibration. Residual ESP32
ADC nonlinearity can require a lookup table in a later model; inspect calibration
residuals before treating counts as a physical voltage measurement.

## Fit a provided paired calibration

Prepare a CSV measured at the ADC input—not the piezo before an unknown
amplifier—with at least two distinct voltages:

```csv
adc_input_voltage_v,raw_adc
<known voltage>,<observed integer count>
<known voltage>,<observed integer count>
```

Then run, replacing the resolution and attenuation with firmware values:

```bash
python -m tass.adc --input calibration/adc_pairs.csv \
  --output calibration/esp32_adc.json \
  --resolution-bits <firmware ADC bits> \
  --attenuation-db <firmware attenuation dB>
```

This fits `count = zero_voltage_code + gain * adc_input_voltage_v` and records
RMSE and maximum residual in counts. It does not identify frequency response,
front-end gain before the calibration point, noise, or sampling jitter.

Generate a new dataset using that saved mapping:

```bash
python -m tass.dataset \
  --config tass_minimum/config/default.json \
  --protocol tass_minimum/config/experiments/dataset_channels_v2.json \
  --adc-config calibration/esp32_adc.json \
  --output outputs/dataset_esp32_adc
```

Alternatively, pass the mapping directly to dataset generation without creating
a calibration JSON. This example is deliberately marked **assumed** and its
numbers are not hardware calibration values:

```bash
python -m tass.dataset \
  --config tass_minimum/config/default.json \
  --protocol tass_minimum/config/experiments/dataset_channels_v2.json \
  --output outputs/dataset_esp32_adc_assumed \
  --adc-resolution-bits 12 \
  --adc-counts-per-volt 1000 \
  --adc-zero-code 2048 \
  --adc-attenuation-db 11
```

Replace all four numbers with firmware/calibration values. Inline conversion is
saved with `parameter_source="assumed"`; its counts are only an approximate
domain match. Optional inline flags configure analog gain/offset, voltage/code
noise, and first-order high/low-pass cutoffs. The three mapping arguments
`--adc-resolution-bits`, `--adc-counts-per-volt`, and `--adc-zero-code` are
required together, preventing accidental partial configuration.

Then extract the same representation used by hardware:

```bash
python -m tass.ml_cli extract --input outputs/dataset_esp32_adc \
  --output outputs/features_esp32_adc \
  --sensor-representation raw_adc --sensor-id 1 --sensor-id 2
```

## Schema and illustrative rows

The generated `example_trial.csv` retains voltage diagnostics and separates
targets from ADC features. Its relevant columns look like:

```csv
time_s,contact_force_N,contact_location_m,string_tension_N,load_cell_readout_N,piezo_sensor_1_V,piezo_sensor_1_raw_adc
0.000000,2.03,0.100,1.88,1.88,0.000000,2048
0.000021,2.03,0.100,1.88,1.88,0.000500,2049
0.000042,2.03,0.100,1.88,1.88,-0.000500,2047
```

Those three numbers are **format-only illustrative rows**, not ESP32
measurements or recommended calibration values. Actual counts are generated only
from the supplied ADC configuration. `load_cell_readout_N` is currently an ideal
alias of simulated tendon tension; no load-cell electronics/noise is modeled.
