import os
# import platform
# print(platform.system())
# print(platform.release())

principal = '/home/joao/Documentos/ADM/ADM'
os.chdir(principal)


chamada25 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python Merge_unstructured_6.py\"'
chamada26 = 'sudo docker run -it -v $PWD:/elliptic padmec/pymoab-pytrilinos:3.6 bash -c \"cd /elliptic; python ams_prol.py\"'

l1 = [chamada25, chamada26]


"""
sudo docker run -it -v $PWD:/elliptic presto bash -c "cd /elliptic; python -m elliptic.Preprocess structured.cfg"
sudo docker pull padmec/elliptic:1.0
sudo docker pull padmec/pymoab-pytrilinos:3.6

"""

# for i in l1:
#     os.system(i)

os.system(chamada26)



# os.system(chamada5)

# caminho_visit = '/home/joao/programas/visit2_10_0.linux-x86_64/bin'
# os.chdir(caminho_visit)
# os.system('./visit')
