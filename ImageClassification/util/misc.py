import csv
import os


class CSVLogger:
    def __init__(self, args, fieldnames, filename='log.csv'):
        self.filename = filename
        self.csv_file = open(filename, 'a')
        writer = csv.writer(self.csv_file)
        if not os.path.exists(filename):
            # Write model configuration at top of csv
            for arg in vars(args):
                writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()
