import http from 'http';

const PORT = 5000;

const server = http.createServer((req, res) => {
    // res.setHeader('Content-Type', 'text/html');
    // res.statusCode = 404;
    res.writeHead(500, { 'Content-Type': 'application/json' })
    // res.end('<h1>Hello World</h1>');
    res.end(JSON.stringify({ message: 'Server Error' }));
});

server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`)
});