
#include "VarLogger.h"
#include <ArduinoSTL.h>
#include <vector>
#include <map>

using namespace arduino;

// Forward declaration
class VarLogger;

const char VarLogger::log0[] PROGMEM = "log0";
const char VarLogger::trace0[] PROGMEM = "trace0";

long VarLogger::created_timestamp;
std::vector<std::pair<String, long>> VarLogger::data;
std::map<String, std::vector<long>> VarLogger::data_dict;
int VarLogger::catchpop = 0;
int VarLogger::write_count = 0;
std::map<String, String> VarLogger::thread_map;
String VarLogger::write_name = log0;
String VarLogger::trace_name = trace0;
std::map<String, String> VarLogger::threads_info;

void VarLogger::initialize() {
  created_timestamp = millis();
  auto files = check_files();
  write_name = files.first;
  trace_name = files.second;
}

void VarLogger::log(String var, String fun, String clas, String th) {
  auto dict_iter = data_dict.find(th + "_" + clas + "_" + fun + "_" + var);
  long log_time = millis() - created_timestamp;

  if (dict_iter != data_dict.end()) {
    auto& var_list = dict_iter->second;

    while (var_list.size() >= 1000) {
      catchpop = var_list.front();
      var_list.erase(var_list.begin());
    }

    var_list.push_back(log_time);
  } else {
    data_dict[th + "_" + clas + "_" + fun + "_" + var] = {log_time};
  }

  log_seq(th + "_" + clas + "_" + fun + "_" + var, log_time);

  write_count++;
  if (write_count >= 10) {
    write_count = 0;
    write_data();
  }
}

void VarLogger::log_seq(String event, long log_time) {
  data.push_back(std::make_pair(event, log_time));
}

void VarLogger::write_data() {
  // Serialize data_dict to a JSON string
  String jsonString;
  serializeJson(data_dict, jsonString);

  // Write the JSON string to flash memory
  write_flash(jsonString.c_str(), write_name.c_str());

  // Similarly, write the trace data to flash memory
  jsonString = ""; // Reset jsonString
  serializeJson(data, jsonString);
  write_flash(jsonString.c_str(), trace_name.c_str());
}

void VarLogger::write_flash(const char* data, const char* filename) {
  // Calculate the size of the data
  size_t dataSize = strlen(data);

  // Open a flash memory space for writing
  PROGMEM const char* flashStart = nullptr;
  uint8_t* writePtr = nullptr;
  for (size_t i = 0; i < dataSize; ++i) {
    writePtr = (uint8_t*)pgm_get_far_address(flashStart + i);
    *writePtr = pgm_read_byte_near(data + i);
  }

  // Store the filename in flash memory
  writePtr = (uint8_t*)pgm_get_far_address(filename);
  for (size_t i = 0; i < strlen(filename) + 1; ++i) {
    *writePtr = pgm_read_byte_near(filename + i);
    ++writePtr;
  }
}

void VarLogger::save() {
  write_data();
}

std::pair<String, String> VarLogger::thread_status(String thread_id, String status) {
  auto ids = threads_info.begin();
  assert(status == "dead" || status == "active" || status == "");

  if (status != "" && thread_id != "") {
    threads_info[thread_id] = status;

    auto num = thread_map.begin();
    if (thread_map.find(thread_id) == num) {
      thread_map[thread_id] = String(thread_map.size());
    }
  } else {
    return std::make_pair(ids->first, threads_info[ids->first]);
  }
}

String VarLogger::map_thread(String thread_id) {
  if (thread_map.find(thread_id) == thread_map.end()) {
    Serial.println("Thread not found");
  }

  return thread_map[thread_id];
}

void VarLogger::traceback(std::exception exc) {
  // Handle traceback in PROGMEM if needed.
}