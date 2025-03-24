# Neuronal Net
Ein GUI das handgeschriebene Zahlen erkennt. 

# Vorausetzung
Python muss installiert sein.

# Ausführung
Das GUI mit dem Neuronal-net kann mit den folgenden Befehla ausgeführt werden
```bash
# Virtuelles Env kreeiren 
python -m venv env

# Env aktivieren (MacOS)
source env/bin/activate

# Env aktvieren (Windows)
.\env\Scripts\activate

# Abhängikeiten installieren
pip install -r requirements.txt

# Zuerst das NEURONAL_NETWORK_SGD.py ausführen 
python ./NEURONAL_NETWORK_SGD.py

# Danach das GUI.py ausführen
python ./GUI.py

# Um eine bessere Performance zu haben das CNN model ausführen
python ./NEURONAL_NETWORK_CNN
```