"""
Universal Data & Document Ingestion Parser.

Extracts biometric features and datasets from:
- PDF medical/athletic reports & research papers (via pypdf)
- Tabular files (CSV, Excel .xlsx, Parquet)
- Raw text/JSON interval streams
"""

import io
import json
import os
import re
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from pypdf import PdfReader

from ml.feature_extractor import BiometricFeatureExtractor


class UniversalDataParser:
    """Parses diverse file formats (PDF, CSV, Excel, JSON) into ML-ready datasets."""

    @classmethod
    def parse_file(cls, file_path_or_bytes: Union[str, bytes], filename: str) -> pd.DataFrame:
        """
        Ingest a file by extension and convert to a pandas DataFrame
        with required ML features or extractable pulse values.
        """
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".csv":
            if isinstance(file_path_or_bytes, bytes):
                return pd.read_csv(io.BytesIO(file_path_or_bytes))
            return pd.read_csv(file_path_or_bytes)

        elif ext in [".xlsx", ".xls"]:
            if isinstance(file_path_or_bytes, bytes):
                return pd.read_excel(io.BytesIO(file_path_or_bytes))
            return pd.read_excel(file_path_or_bytes)

        elif ext == ".parquet":
            if isinstance(file_path_or_bytes, bytes):
                return pd.read_parquet(io.BytesIO(file_path_or_bytes))
            return pd.read_parquet(file_path_or_bytes)

        elif ext == ".json":
            if isinstance(file_path_or_bytes, bytes):
                data = json.loads(file_path_or_bytes.decode("utf-8"))
            else:
                with open(file_path_or_bytes, "r") as f:
                    data = json.load(f)

            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Check if it contains records under a key
                for k in ["data", "records", "samples", "rows"]:
                    if k in data and isinstance(data[k], list):
                        return pd.DataFrame(data[k])
                return pd.DataFrame([data])

        elif ext == ".pdf":
            return cls.parse_pdf(file_path_or_bytes)

        elif ext in [".txt", ".log"]:
            return cls.parse_raw_text(file_path_or_bytes)

        else:
            raise ValueError(f"Unsupported file format '{ext}'. Expected .csv, .xlsx, .json, .parquet, or .pdf.")

    @classmethod
    def parse_pdf(cls, file_path_or_bytes: Union[str, bytes]) -> pd.DataFrame:
        """
        Extract numerical biometric records and tables from PDF pages.
        Extracts numbers, table rows, and key-value pairs (e.g. HR, RMSSD, SDNN, Label).
        """
        if isinstance(file_path_or_bytes, bytes):
            reader = PdfReader(io.BytesIO(file_path_or_bytes))
        else:
            reader = PdfReader(file_path_or_bytes)

        full_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

        joined_text = "\n".join(full_text)

        # 1. Try to find tabular numbers / CSV-like patterns in text
        records = cls._extract_records_from_text(joined_text)
        if len(records) > 0:
            df = pd.DataFrame(records)
            return cls._standardize_dataframe(df)

        raise ValueError(
            "Could not detect structured biometric data or tables in the provided PDF. "
            "Ensure the PDF contains numerical measurements, HRV metrics, or tabular records."
        )

    @classmethod
    def parse_raw_text(cls, file_path_or_bytes: Union[str, bytes]) -> pd.DataFrame:
        """Parse raw text file containing comma/space/newline separated values or JSON."""
        if isinstance(file_path_or_bytes, bytes):
            content = file_path_or_bytes.decode("utf-8", errors="ignore")
        else:
            with open(file_path_or_bytes, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        # Check if content is JSON
        content_stripped = content.strip()
        if (content_stripped.startswith("{") and content_stripped.endswith("}")) or \
           (content_stripped.startswith("[") and content_stripped.endswith("]")):
            try:
                data = json.loads(content_stripped)
                if isinstance(data, list):
                    return pd.DataFrame(data)
                return pd.DataFrame([data])
            except Exception:
                pass

        # Parse numbers from text
        records = cls._extract_records_from_text(content)
        if records:
            return pd.DataFrame(records)

        # Fallback: single column of RR intervals
        numbers = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", content)]
        if len(numbers) >= 10:
            return pd.DataFrame({"rr_interval_ms": numbers})

        raise ValueError("Could not parse valid numerical records from text file.")

    @classmethod
    def _extract_records_from_text(cls, text: str) -> List[Dict[str, Any]]:
        """Extract key-value pairs and tabular rows from text."""
        records = []
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        current_record = {}
        for line in lines:
            # Check for key: value pairs like "HR: 65, RMSSD: 52"
            matches = re.findall(r"([A-Za-z_]+)\s*[:=]\s*([0-9\.\-]+)", line)
            if matches:
                for k, v in matches:
                    k_clean = k.lower().strip()
                    try:
                        current_record[k_clean] = float(v)
                    except ValueError:
                        pass
                if len(current_record) >= 2:
                    records.append(current_record.copy())
                    current_record = {}
                continue

            # Check for comma-separated or tab-separated numbers
            tokens = re.split(r"[\t,;]+", line)
            if len(tokens) >= 3:
                try:
                    num_row = [float(t.strip()) for t in tokens if t.strip()]
                    if len(num_row) >= 3:
                        records.append({f"feature_{i}": val for i, val in enumerate(num_row)})
                except ValueError:
                    pass

        return records

    @classmethod
    def _standardize_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Map common column variations to standardized feature names."""
        column_synonyms = {
            "heart_rate": "hr_mean",
            "hr": "hr_mean",
            "pulse": "hr_mean",
            "rr": "rr_mean",
            "mean_rr": "rr_mean",
            "rmssd_ms": "rmssd",
            "sdnn_ms": "sdnn",
            "tier": "recovery_tier",
            "class": "label",
            "target": "label",
        }
        df = df.rename(columns={col: column_synonyms[col.lower()] for col in df.columns if col.lower() in column_synonyms})

        # If string labels (GREEN, YELLOW, RED), map to numerical
        if "label" not in df.columns and "recovery_tier" in df.columns:
            tier_map = {"GREEN": 0, "YELLOW": 1, "RED": 2}
            df["label"] = df["recovery_tier"].str.upper().map(tier_map)

        return df
