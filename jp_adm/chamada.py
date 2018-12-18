import os
# import platform
# print(platform.system())
# print(platform.release())

principal = '/home/joao/Documentos/ADM/ADM/jp_adm/'
os.chdir(principal)


chamada25 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python Merge_unstructured_6.py\"'
chamada26 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python ams_prol.py\"'
'sudo docker run -it -v  $PWD:/pytest desenvolvimento:latest bash -c "cd /pytest; bash"'
'docker run -it -v  $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c "cd /elliptic; bash"'
# 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; bash"

limpar_cache = ['sudo sync', 'sudo sysctl -w vm.drop_caches=3', 'sudo sysctl -w vm.drop_caches=0']
# limpar_cache2 = ['sudo echo 3 > /proc/sys/vm/drop_caches', 'sudo sysctl -w vm.drop_caches=3']


l1 = [chamada25, chamada26]


"""
sudo docker run -it -v $PWD:/elliptic presto bash -c "cd /elliptic; python -m elliptic.Preprocess structured.cfg"
sudo docker pull padmec/elliptic:1.0
sudo docker pull padmec/pymoab-pytrilinos:3.6

"""

# for i in limpar_cache:
#     os.system(i)

# for i in l1:
#     os.system(i)

os.system(chamada26)

# for i in limpar_cache:
#     os.system(i)




# os.system(chamada5)

# caminho_visit = '/home/joao/programas/visit2_10_0.linux-x86_64/bin'
# os.chdir(caminho_visit)
# os.system('./visit')
