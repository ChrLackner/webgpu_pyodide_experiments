rm -f *.tar.bz2
wget https://ngsolve.org/files/ngsolve_pyodide_0.26.2.tar.bz2
tar xvf ngsolve_pyodide_0.26.2.tar.bz2
cd pyodide
ln -s ../index.html
ln -s ./pyodide.js
ln -s ../code.py
