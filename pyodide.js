 async function main(){
    let pyodide = await loadPyodide();
     pyodide.setDebug(true);
    const python_code = await((await fetch("./code.py")).text());
    pyodide.runPythonAsync(python_code);
}
main();
