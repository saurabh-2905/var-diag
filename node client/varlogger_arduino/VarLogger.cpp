#include "VarLogger.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <sstream>
#include <ArduinoJson.h>
#include <SD.h>
#include <SPI.h>
#include <mutex>
#include <map>
#include <vector>
#include <utility>

int VarLogger::buffer_select = 1;
int VarLogger::save_buffer = 0;
int VarLogger::buffer_index = 0;
unsigned long VarLogger::created_timestamp = millis();
unsigned long VarLogger::time_to_write = 0;
std::map<int, std::vector<std::pair<unsigned long, int>>> VarLogger::data_dict;
std::map<std::string, int> VarLogger::_vardict;
std::map<std::string, int> VarLogger::_thread_map;
std::map<std::string, std::string> VarLogger::threads_info;

std::vector<std::pair<int, unsigned long>> VarLogger::data1(TRACE_LENGTH, {0, 0});
std::vector<std::pair<int, unsigned long>> VarLogger::data2(TRACE_LENGTH, {0, 0});

int VarLogger::prev1_event = -1;
int VarLogger::prev2_event = -1;
unsigned long VarLogger::prev1_time = 0;
unsigned long VarLogger::prev2_time = 0;
int VarLogger::_write_count = 0;
std::string VarLogger::write_name = "log0";
std::string VarLogger::trace_name = "trace0";
int VarLogger::cur_file = 0;
bool VarLogger::sdInitialized = false;

bool VarLogger::initializeSDCard(int csPin) {
    if (!SD.begin(csPin)) {
        Serial.println("SD Card initialization failed!");
        sdInitialized = false;
        return false;
    }
    Serial.println("SD Card initialized successfully.");
    sdInitialized = true;
    return true;
}

void VarLogger::log(const char* var, const char* fun, const char* clas, const char* th, int val, bool save) {
    unsigned long log_time = millis() - created_timestamp - time_to_write;
    std::string event_key = std::string(th) + "-" + std::string(clas) + "-" + std::string(fun) + "-" + std::string(var);
    int event_num = var2int(event_key);

    if (prev1_event != event_num) {
        log_seq(event_num, log_time);
        _write_count++;
    }

    if (_write_count >= TRACE_LENGTH && save != false) {
        _write_count = 0;
        unsigned long start_time = millis();
        write_data();
        time_to_write += millis() - start_time;
        data1.clear();
    }
    prev2_event = prev1_event;
    prev1_event = event_num;
    prev2_time = prev1_time;
    prev1_time = log_time;
}

int VarLogger::var2int(const std::string& var) {
    if (_vardict.find(var) == _vardict.end()) {
        _vardict[var] = _vardict.size();
    }
    return _vardict[var];
}

std::string VarLogger::int2var(int num) {
    for (auto &it : _vardict) {
        if (it.second == num) {
            return it.first;
        }
    }
    return "";
}

void VarLogger::log_seq(int event, unsigned long log_time) {
    if (buffer_select == 1) {
        if (buffer_index == TRACE_LENGTH - 1) {
            data1[buffer_index] = {event, log_time};
            buffer_select = 2;
            save_buffer = 1;
            buffer_index = 0;
        } else {
            data1[buffer_index++] = {event, log_time};
        }
    } else {
        if (buffer_index == TRACE_LENGTH - 1) {
            data2[buffer_index] = {event, log_time};
            buffer_select = 1;
            save_buffer = 2;
            buffer_index = 0;
        } else {
            data2[buffer_index++] = {event, log_time};
        }
    }
}

void VarLogger::generateFileNames() {
  if(!sdInitialized) {
    Serial.println("SD Card not initialized.");
    return;
  }
  File counterFile = SD.open("/filecounter.txt",FILE_READ);
  if(counterFile) {
    cur_file = counterFile.parseInt(); //For reading the current filenumber
    counterFile.close();
  } else {
    cur_file = 0;
  }

  trace_name = "trace" + std::to_string(cur_file);
  write_name = "log" + std::to_string(cur_file);
  cur_file++;
  counterFile = SD.open("/filecounter.txt",FILE_WRITE);
  if(counterFile) {
    counterFile.println(cur_file);
    counterFile.close();
  } else {
    Serial.println("Error opening filecounter.txt for writing");
  }
}


void VarLogger::write_data() {
    if (!sdInitialized) {
        Serial.println("SD Card not initialized!");
        return;
    }

    int retries = 3;  
    bool writeSuccess = false;
    std::string traceFilePath = "/"+ trace_name+".txt";
    const char* traceFilePath_F = traceFilePath.c_str();
    while(retries > 0 && !writeSuccess) {
      File traceFile = SD.open(traceFilePath_F, FILE_WRITE);
      if (traceFile) {
          StaticJsonDocument<1024> jsonDoc;
          if (save_buffer == 1) {
              for (auto &entry : data1) {
                  JsonArray array = jsonDoc.createNestedArray();
                  array.add(entry.first);
                  array.add(entry.second);
              }
              serializeJson(jsonDoc, traceFile);
              save_buffer = 0;
          } else if (save_buffer == 2) {
              for (auto &entry : data2) {
                  JsonArray array = jsonDoc.createNestedArray();
                  array.add(entry.first);
                  array.add(entry.second);
              }
              serializeJson(jsonDoc, traceFile);
              save_buffer = 0;
          }
          traceFile.close();
          writeSuccess = true;
      } else {
        Serial.println("Error writing trace file, retrying...");
        retries--;
        delay(100);
      }
    }

    if(!writeSuccess) {
      Serial.println("Failed to write trace file after 3 attempts");
      return;
    }

    retries = 3;
    writeSuccess = false;
    std::string varFilePath = "/varlist" + std::to_string(cur_file) + ".txt";
    const char* varFilePathCStr = varFilePath.c_str();

    while(retries > 0 && !writeSuccess) {
      File varFile = SD.open(varFilePathCStr, FILE_WRITE);
      if (varFile) {
          StaticJsonDocument<1024> jsonDoc;
          for (auto &entry : _vardict) {
              jsonDoc[entry.first] = entry.second;
          }
          serializeJson(jsonDoc, varFile);
          varFile.close();
          writeSuccess = true;
      } else {
        Serial.println("Error writing varlist file, retrying...");
        retries--;
        delay(100);
      }
    }

    if(!writeSuccess) {
      Serial.println("Failed to write varlist file after 3 attempts.");
      return;
    }
    cur_file++;
    generateFileNames(); //For getting the next trace and log file names
}

void VarLogger::save() {
    write_data();
}

void VarLogger::threadStatus(const std::string& thread_id, const std::string& status) {
    if (status != "" && thread_id != "") {
        threads_info[thread_id] = status;
        if (_thread_map.find(thread_id) == _thread_map.end()) {
            _thread_map[thread_id] = _thread_map.size();
        }
    }
}

int VarLogger::mapThread(const std::string& thread_id) {
    if (_thread_map.find(thread_id) == _thread_map.end()) {
        return -1; 
    }
    return _thread_map[thread_id];
}

void VarLogger::traceback(const std::string& exc) {
    if (!SD.begin()) {
        return;
    }

    File traceFile = SD.open("/traceback.txt", FILE_WRITE);
    if (traceFile) {
      std::string message = std::to_string(millis()) + ": Exception - " + exc;
      traceFile.println(message.c_str());
      traceFile.close();
    }
}

