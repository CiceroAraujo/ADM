import os

# os.system('sudo docker run -it -v  $PWD:/pytest desenvolvimento:latest bash -c "cd /pytest/output; bash"')
# os.system('sudo docker run -it -v  $PWD:/pytest desenvolvimento:latest bash -c "cd /pytest; bash"')
os.system('sudo docker run -it -v  $PWD:/pytest desenvolvimento-scipy:latest bash -c "cd /pytest; bash"')
