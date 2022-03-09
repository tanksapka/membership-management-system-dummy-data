# Dummy data generator
> Script suite to generate Hungarian dummy data for membership management system.

# Usage
1. Install Python (3.7.9 or higher)
2. Create virtual environment `python -m venv venv`
3. Activate virtual environment `venv\Scripts\activate.bat` or `source venv/bin/activate`
4. Install required modules from requirements.txt `pip install -r requirements.txt`
5. Run main.py file which contains a simple CLI `pyhton main.py -h`
6. Example: `python main.py -o output -t xls -i uuid`

# Sources
- First names
  - http://www.nytud.mta.hu/oszt/nyelvmuvelo/utonevek/index.html
- Last names
  - https://www.nyilvantarto.hu/hu/statisztikak ("Lakossági számadatok")
  - http://vagyok.net
- Zip codes, settlements
  - https://github.com/tamas-ferenci/IrszHnk/blob/master/IrszHnk.csv
- Street names
  - https://data2.openstreetmap.hu/utcalista.php
- Landline phone area codes
  - http://korzetszam.keresok.info/korzetszamok
- Mobile phone codes
  - https://hu.wikipedia.org/wiki/K%C3%B6rzeth%C3%ADv%C3%B3sz%C3%A
  - https://nmhh.hu/dokumentum/220055/NMHH_mobilpiaci_jelentes_2017Q1_2020Q4.pdf
