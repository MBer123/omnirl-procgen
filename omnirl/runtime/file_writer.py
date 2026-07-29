import csv
import os


class _CSVStream:
    """
    Simple append CSV stream.

    Args:
        path: File path of the CSV file to write.
    """

    def __init__(self, path: str):
        self.path = path
        self.fieldnames = None
        self.writer = None

        file_exists = os.path.exists(path)
        file_nonempty = file_exists and os.path.getsize(path) > 0

        if file_nonempty:
            with open(path, "r", newline="", encoding="utf-8") as fp:
                reader = csv.DictReader(fp)
                self.fieldnames = list(reader.fieldnames or [])

        self.fp = open(path, "a", newline="", encoding="utf-8")

        if self.fieldnames:
            self.writer = csv.DictWriter(
                self.fp,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
            )

    def write(self, row):
        """Write one row of metrics into the CSV file."""
        if not row:
            return

        clean_row = {k: v for k, v in row.items()}

        if self.fieldnames is None:
            self.fieldnames = list(clean_row.keys())
            self.writer = csv.DictWriter(
                self.fp,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
            )
            self.writer.writeheader()

        self.writer.writerow({k: clean_row.get(k, "") for k in self.fieldnames})
        self.fp.flush()

    def close(self):
        """Close the underlying CSV file stream."""
        self.fp.close()


class FileWriter:
    """
    Metrics CSV writer.

    Args:
        ckpt_dir: Directory used to save metric CSV files.
        log_name: File name of the metrics CSV.
    """

    def __init__(
        self,
        ckpt_dir: str,
        log_name: str = "log.csv",
    ):
        os.makedirs(ckpt_dir, exist_ok=True)

        self.log_stream = _CSVStream(os.path.join(ckpt_dir, log_name))

    def write(self, row):
        """Write one row of metrics into csv."""
        self.log_stream.write(row)

    def close(self):
        """Close csv file streams."""
        self.log_stream.close()
