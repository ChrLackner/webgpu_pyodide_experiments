 async function main(){
    let pyodide = await loadPyodide();
     pyodide.setDebug(true);
    // await pyodide.loadPackage(['netgen']);
    const python_code = await((await fetch("./code.py", {method: "GET", cache: "no-cache"})).text());
    pyodide.runPythonAsync(python_code);
}
main();
