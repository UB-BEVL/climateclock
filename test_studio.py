from pathlib import Path
import base64
import io
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]/'climateclock'
sys.path.insert(0,str(ROOT))
import psychrometric_studio as studio
from PIL import Image

class StudioTests(unittest.TestCase):
    def fixture(self):
        png=io.BytesIO(); Image.new('RGB',(800,450),'white').save(png,format='PNG')
        return {'units':'SI','meta':{'name':'Test cooling case'}, 'pressure':'101.325 kPa',
                'labels':{'temperature':'°C','duty':'kW','airflow':'L/s','massFlow':'kg/s','moistureRate':'kg/h'},
                'statePoints':[{'point':1,'name':'Outdoor air','tdb':25.,'twb':18.,'tdp':14.,'rh':.5,'w':10.,'h':50.,'v':.85,'massFlow':1.2,'airflow':1000.}],
                'loads':[{'point':2,'name':'Cooling coil','total':-10.,'sensible':-8.,'latent':-2.,'shr':.8,'moisture':-1.,'adp':10.,'bypass':.1}],
                'totals':{'cooling':-10.,'heating':0.,'balance':'Energy balance closes.'},
                'chartPng':base64.b64encode(png.getvalue()).decode()}
    def test_real_pdf_builder_accepts_both_cases(self):
        case=self.fixture(); pdf=studio.build_studio_pdf([case,{**case,'meta':{'name':'Test heating case'}}])
        self.assertTrue(pdf.startswith(b'%PDF-'))
        self.assertGreater(len(pdf),3000)
    def test_rejects_invalid_image_and_units(self):
        for case in [{**self.fixture(),'chartPng':'invalid'}, {**self.fixture(),'units':'invalid'}]:
            with self.assertRaises(Exception): studio.validate_cases([case])
    def test_weather_urls_remain_restricted(self):
        valid='https://climate.onebuilding.org/WMO_Region_4/test.zip'
        self.assertEqual(studio._allowed_weather_url(valid),valid)
        for url in ['http://climate.onebuilding.org/test.zip','https://example.com/test.zip','https://climate.onebuilding.org.evil.test/test.zip','https://climate.onebuilding.org:8000/test.zip','https://user@climate.onebuilding.org/test.zip']:
            with self.assertRaises(ValueError): studio._allowed_weather_url(url)
    def test_report_from_previous_station_is_excluded(self):
        state={'last_parsed_epw_hash':'new','psychrometric_studio_report_source':'old','psychrometric_studio_report':[self.fixture()]}
        with patch.object(studio.st,'session_state',state):
            self.assertEqual(studio.current_report_cases(),[])
            state['psychrometric_studio_report_source']='new'
            self.assertEqual(len(studio.current_report_cases()),1)

    def test_normalized_weather_has_epw_hours_and_station_metadata(self):
        import csv
        import pandas as pd
        frame = pd.DataFrame({'drybulb':[12.,15.], 'relhum':[50.,60.], 'atmos_pressure':[100000.,100000.]}, index=pd.to_datetime(['2021-01-01 00:00','2021-12-31 23:00']))
        rows = list(csv.reader(io.StringIO(studio.weather_text_from_frame(frame, {'location':{'city':'Test, City','latitude':42.}}))))
        self.assertEqual(len(rows),10)
        self.assertEqual(rows[0][1],'Test  City')
        self.assertEqual(rows[8][0:5],['2021','1','1','1','60'])
        self.assertEqual(rows[9][3],'24')
        self.assertEqual(len(rows[9]),35)
        self.assertEqual(rows[8][6:10],['12.0','99.9','50.0','100000.0'])

if __name__=='__main__': unittest.main(verbosity=2)
