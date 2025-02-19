import time
import threading
import json

class Logger:

    ## Initialize the logger with two buffers, a flush interval, and a buffer size
    def __init__(self, buffer_size=100, flush_interval=5):
        self.bufferA = []
        self.bufferB = []
        self.active_buffer = self.bufferA
        self.inactive_buffer = self.bufferB
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.lock = threading.Lock()
        self.timer = threading.Timer(flush_interval, self.flush_to_storage)
        self.start_timer()

        # Define event schema
        self.event_schema = {
            "eventId": int,
            "timestamp": float,
            "data": dict,
        }

    ## Start the periodic timer for flushing
    def start_timer(self):
        self.timer = threading.Timer(self.flush_interval, self.flush_to_storage)
        self.timer.start()

    ## Log an event with dynamic attributes
    ## event_id: Identifier for the type of event
    ## attributes: Additional data associated with the event
    def log_event(self, event_id, **attributes):
        timestamp = time.time()
        event = {
            'eventId': event_id,
            'timestamp': timestamp,
            'data': attributes
        }

        # Validate event against schema
        self.clean_data(event)

        with self.lock:
            self.active_buffer.append(event)
            if len(self.active_buffer) >= self.buffer_size:
                self.flush_to_storage()

    ## Ensures the event matches the defined schema
    def clean_data(self, event):
        for key, data_type in self.event_schema.items():
            if key not in event or not isinstance(event[key], data_type):
                raise ValueError(f"Event does not match schema: {event}")

    ## Flush the current buffer to storage and swap buffers
    def flush_to_storage(self):
        with self.lock:
            self.active_buffer, self.inactive_buffer = self.inactive_buffer, self.active_buffer       # Swap buffers

        self.write_to_flash(self.inactive_buffer)   # Write the inactive buffer to storage
        self.inactive_buffer.clear()                #clearing the buffer
        self.start_timer()                          # Restart the timer

    ## Simulate writing the buffer to flash storage
    def write_to_flash(self, buffer):
        print(f"Flushing {len(buffer)} events to storage.")
        for event in buffer:
            print(f"Stored Event: {event}")

    ## Label events based on a user-defined predicate.
    ## buffer: The buffer to be labeled.
    ## dictionary with labeled "good" and "bad" events.
    def label_data(self, buffer):
        labeled_data = {"good": [], "bad": []}
        for event in buffer:
            if self.is_good_event(event):
                labeled_data["good"].append(event)
            else:
                labeled_data["bad"].append(event)
        return labeled_data

    ## Function to classify the events
    def is_good_event(self, event):
        return event["data"].get("key") != "bad_value"

    ## Convert labeled data into a specific analysis format.
    ## labeled_data: Dictionary with labeled data.
    ## format_type: Desired format ("json" or "csv").
    ## Formatted data as a string.
    def convert_to_analysis_format(self, labeled_data, format_type="json"):
        if format_type == "json":
            return json.dumps(labeled_data, indent=4)
        elif format_type == "csv":
            import csv
            from io import StringIO

            output = StringIO()
            csv_writer = csv.writer(output)
            csv_writer.writerow(["Label", "EventId", "Timestamp", "Data"])
            for label, events in labeled_data.items():
                for event in events:
                    csv_writer.writerow([label, event['eventId'], event['timestamp'], event['data']])
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    ## Stop the logger and flush remaining data
    def stop(self):
        self.timer.cancel()
        self.flush_to_storage()


# Example Usage
if __name__ == "__main__":
    # Initialize the Logger
    logger = Logger(buffer_size=5, flush_interval=10)

    # Simulate logging events with dynamic attributes
    logger.log_event(1, key="value1", status="active")
    logger.log_event(2, key="value2", status="error")
    logger.log_event(3, key="value3", status="active")
    logger.log_event(4, key="bad_value", status="error")
    logger.log_event(5, key="value5", status="active")  # This triggers an immediate flush

    time.sleep(12)  # Let the timer trigger another flush

    # Add more logs
    logger.log_event(6, key="value6", status="active")

    # Stop logging
    logger.stop()

    # Process logged data
    labeled_data = logger.label_data(logger.inactive_buffer)
    converted_data_json = logger.convert_to_analysis_format(labeled_data, format_type="json")
    converted_data_csv = logger.convert_to_analysis_format(labeled_data, format_type="csv")

    # Output results
    print("\nLabeled Data in JSON format:")
    print(converted_data_json)

    print("\nLabeled Data in CSV format:")
    print(converted_data_csv)
