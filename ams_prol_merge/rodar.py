import os

rodar = 'docker run -t -it -v /home/j/Área\ de\ trabalho/TPFA:/elliptic_case padmec/elliptic:1.0 bash -c "cd /elliptic_case;python3 ams_prol.py"'
os.system(rodar)
