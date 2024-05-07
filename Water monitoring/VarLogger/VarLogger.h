#ifndef VARLOGGER_H
#define VARLOGGER_H

#include "Arduino.h"
#include <stdexcept>

class VarLogger {
public:
  static void initialize();
  static void log(String var, String fun, String clas, String th);
  static void log_seq(String event, long log_time);
  static void write_data();
  static void save();
  static void traceback(std::exception exc);
  static std::pair<String, String> thread_status(String thread_id, String status);
  static String map_thread(String thread_id);

private:
  static long created_timestamp;
  static std::vector<std::pair<String, long>> data;
  static std::map<String, std::vector<long>> data_dict;
  static int catchpop;
  static int write_count;
  static std::map<String, String> thread_map;
  static String write_name;
  static String trace_name;
  static std::map<String, String> threads_info;
  static const char log0[] PROGMEM;
  static const char trace0[] PROGMEM;

  static void write_flash(const char* data, const char* filename);
  static std::pair<String, String> check_files();
};

#endif // VARLOGGER_H
