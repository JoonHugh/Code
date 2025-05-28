import express from 'express';
import { fileURLToPath } from 'url';
import path from 'path';
import routes from './routes/routes.js'
import logger from './middleware/logger.js';


const PORT = process.env.PORT || 5000; 

const app = express();

app.use(logger);

app.use(express.json());
app.use(express.urlencoded({ extended: false }));


const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
console.log(__filename, __dirname);


app.use(express.static(path.join(__dirname, 'html')));
app.use(
    '/images',
    express.static(path.join(__dirname, 'images'))
);

app.use('/user/', routes);


app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`)
})