"""The notebook's first DICOM reader (cell 0). Not used by the pipeline:
data/dicom.py replaced it. Kept for reference."""
import os
from collections import defaultdict
from glob import glob

import pandas as pd
import pydicom


class DICOMExtractor :
    def __init__(self, data_path) :
        self.data_path = data_path

    def _getStudyInstanceUID(self, file = 'train.csv') -> list[str] :
        df = pd.read_csv(os.path.join(self.data_path, file))
        return df['StudyInstanceUID'].to_list()

    def _getSeriesInstanceUID(self, file = 'train_series.csv') -> dict:
        df = pd.read_csv(os.path.join(self.data_path, file))
        seriesInstanceUID = defaultdict(list)
        for study, series in zip(df['StudyInstanceUID'], df['SeriesInstanceUID']) :
            seriesInstanceUID[study].append(series)

        return seriesInstanceUID

    
    def getDICOM(self, metadata_only = False, file = 'train_series'):
        dicomInstances = {}
        for study, series in self._getSeriesInstanceUID().items():
            for ser in series:
                series_dir = os.path.join(self.data_path, file, study, ser)
                paths = sorted(glob(os.path.join(series_dir, "*.dcm")))
                
                if not paths:
                    continue
    
                datasets = []
                for p in paths:
                    try:
                        datasets.append(pydicom.dcmread(p, stop_before_pixels=metadata_only))
                    except Exception as e:
                        print(f"skipping {p}: {type(e).__name__}: {e}")
    
                if not datasets:
                    continue
    
                #iop = np.array(datasets[0].ImageOrientationPatient, float)
                #normal = np.cross(iop[:3], iop[3:])
                #datasets.sort(key=lambda d: float(np.dot(np.array(d.ImagePositionPatient, float), normal)))
                yield datasets
        
