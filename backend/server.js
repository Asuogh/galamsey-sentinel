const express = require('express');
const cors = require('cors');

const app = express();
const PORT = 5000;
 
app.use(cors());
app.use(express.json()); 

app.get('/health', (req, res) => {
    console.log('[GET] /health - Checking server status...');
    res.status(200).json({ status: 'API is running beautifully!' });
});


app.post('/api/predict', (req, res) => {
    const { lat, lon } = req.body;
    
    console.log(`[POST] /api/predict - Received coordinates: Lat ${lat}, Lon ${lon}`);

    // Basic validation
    if (!lat || !lon) {
        return res.status(400).json({ error: 'Latitude and Longitude are required.' });
    }

    // Send back the mock AI prediction
    res.status(200).json({
        status: 'success',
        prediction: 'galamsey',
        confidence: 0.92,
        spm: 182.26
    });
});

// 3. Catch-all for missing routes (Fixes the Express 5 crash)
app.use((req, res) => {
    res.status(404).json({ error: 'Endpoint not found' });
});

// Start the server
app.listen(PORT, () => {
    console.log('=============================================');
    console.log('🚀 GALAMSEY SENTINEL API IS LIVE');
    console.log(`📍 Server running on http://localhost:${PORT}`);
    console.log('=============================================');
});