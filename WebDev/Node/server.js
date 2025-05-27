import http from 'http';
import fs from 'fs/promises';
import url from 'url';
import path from 'path';

const PORT = process.env.PORT;
// Get Current path
const __filename = url.fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log(__filename, __dirname);

const server = http.createServer(async (req, res) => {
    try {

        // Check if GET request
        if (req.method === 'GET') {
            let filePath;
            if (req.url === '/') {
                filePath = path.join(__dirname, 'public', 'index.html')
                // res.writeHead(200, { 'Content-Type': 'text/html' })
                // res.end('<h1>Home Page</h1>');
            } else if (req.url === '/about') {
                filePath = path.join(__dirname, 'public', 'about.html')
                // res.writeHead(200, { 'Content-Type': 'text/html' })
                // res.end('<h1>About</h1>');
            } else {
                throw new Error('Not Found');
                // res.writeHead(404, { 'Content-Type': 'text/html' })
                // res.end('<h1>Not Found</h1>');
            }

            const data = await fs.readFile(filePath);
            res.setHeader('Content-Type', 'text/html');
            res.write(data);
            res.end();
        } else {
            throw new Error('Method not allowed');
        }
    } catch (error) {
        res.writeHead(500, { 'Content-Type': 'text/plain' })
        res.end('Server Error');
    }
    

    // console.log(req.url);
    // console.log(req.method);

    // res.setHeader('Content-Type', 'text/html');
    // res.statusCode = 404;
    // res.end(JSON.stringify({ message: 'Server Error' }));
    
});

// app.post('/user', () => {});

server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`)
});