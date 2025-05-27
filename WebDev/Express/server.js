import express from 'express';
import { fileURLToPath } from 'url';
import path from 'path';
import posts from './routes/posts.js';
import logger from './middleware/logger.js';
import error from './middleware/error.js';
import notFound from './middleware/notFound.js';

const PORT = process.env.PORT || 5000;


const app = express();
// Logger middleware
app.use(logger);

// Body Parser middleware
app.use(express.json());
app.use(express.urlencoded({ extended: false } ));

// Setup static folder
const __filename = fileURLToPath(import.meta.url);
console.log(__filename);
const __dirname = path.dirname(__filename);
console.log(__dirname);
app.use(express.static(path.join(__dirname, 'public')));

// app.get('/', (req, res) => { // respond to a get request 
//     // res.send("Hello World");
//     res.sendFile(path.join(__dirname, 'public', 'index.html'));
// });  // get
// app.get('/about', (req, res) => { // respond to a get request 
//     // res.send("About");
//     res.sendFile(path.join(__dirname, 'public', 'about.html'));
// });  // get


// Routes
app.use('/api/posts', posts);

// catch all handling middleware
app.use(notFound);
// Error Handling middleware
app.use(error);

app.listen(PORT, () => console.log(`Server is running on port ${PORT}`));