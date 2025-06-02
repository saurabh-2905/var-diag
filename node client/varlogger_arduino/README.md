# VarLogger: Lightweight Trace Logging mechanism

`VarLogger` is a C++ logging method that logs **event sequences and timestamps** to an SD card. It captures program variable changes, tracks thread activity, and stores trace data in a structured format (JSON), making it suitable for post-analysis, anomaly detection, or debugging.

---

## 📦 Features

- **Efficient Logging**: 
  - Logs variable changes with their timestamps and metadata (variable name, function, class, thread).
  - Supports double-buffering (`data1`, `data2`) for continuous logging without blocking.

- **Trace Data Storage**:
  - Logs are saved as JSON files on the SD card.
  - Each trace contains `[eventID, timestamp]` pairs.
  - Variable mappings are stored in separate JSON files for easy decoding.

- **SD Write**:
  - Retries SD writes up to 3 times in case of failure.
  - Generates unique filenames for each trace and variable mapping.
  - Maintains a persistent `filecounter.txt` for naming consistency.

- **Traceback Support**:
  - Captures exception messages with timestamps in a dedicated `traceback.txt` file.

- **Customizable Configuration**:
  - Supports configurable trace length (`TRACE_LENGTH`).
  - Lightweight memory usage for resource-constrained devices.

---

## Project Structure
```bash
├── VarLogger.cpp   # VarLogger implementation.
├── Varlogger.h     # Varlogger class definition and member declarations.
```

## Logging functions
- `log(var, fun, clas, th, val, save)`:
  - Logs an event: variable name (`var`), function (`fun`), class (`clas`), thread (`th`), value (`val`), and optional save trigger (`save`).

- `traceback(exc)`:
  - Writes exception information to `traceback.txt`.


### File Management
- `initializeSDCard(csPin)`:
  - Initializes the SD card using the specified Chip Select pin.

- `generateFileNames()`:
  - Determines the next trace and variable file names using `filecounter.txt`.

- `write_data()`:
  - Flushes buffer data to SD card files (`trace.txt` and `varlist.txt`).

- `save()`:
  - Manually triggers data flush.

