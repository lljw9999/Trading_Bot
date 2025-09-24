#!/usr/bin/env python3
"""
Debug Dashboard - Minimal version to test frontend data loading
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import json

app = FastAPI()


@app.get("/api/test")
async def test_api():
    return {"message": "API is working", "btc_price": 116528.84, "eth_price": 3835.49}


@app.get("/")
async def dashboard():
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Debug Dashboard</title>
</head>
<body>
    <h1>Debug Dashboard</h1>
    <div>BTC Price: <span id="btcPrice">Loading...</span></div>
    <div>ETH Price: <span id="ethPrice">Loading...</span></div>
    <div>Status: <span id="status">Starting...</span></div>
    
    <script>
        console.log('JavaScript loaded');
        
        async function testFetch() {
            console.log('Starting fetch test...');
            document.getElementById('status').textContent = 'Fetching...';
            
            try {
                const response = await fetch('/api/test');
                console.log('Response received:', response.status);
                
                if (response.ok) {
                    const data = await response.json();
                    console.log('Data received:', data);
                    
                    document.getElementById('btcPrice').textContent = `$${data.btc_price}`;
                    document.getElementById('ethPrice').textContent = `$${data.eth_price}`;
                    document.getElementById('status').textContent = 'Success!';
                } else {
                    document.getElementById('status').textContent = `Error: ${response.status}`;
                }
            } catch (error) {
                console.error('Fetch error:', error);
                document.getElementById('status').textContent = `Error: ${error.message}`;
            }
        }
        
        document.addEventListener('DOMContentLoaded', () => {
            console.log('DOM loaded, starting test...');
            testFetch();
        });
    </script>
</body>
</html>
    """
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
