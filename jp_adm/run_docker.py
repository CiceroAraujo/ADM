import os

os.system('sudo docker run -it -v  $PWD:/pytest desenvolvimento:latest bash -c "cd /pytest; bash"')
