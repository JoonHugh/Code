import express from 'express';
import { fileURLToPath } from 'url';
import path from 'path';
import colors from 'colors';
import connectDB from './config/db.js';
import routes from './routes/routes.js';
import logger from './middleware/logger.js';
import dotenv from 'dotenv';


const PORT = process.env.PORT || 5000; 

const config = dotenv.config();

connectDB();

const app = express();


app.use(logger);

app.use(express.json());
app.use(express.urlencoded({ extended: false }));


const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
console.log(__filename, __dirname);


app.use(express.static(path.join(__dirname, '../frontend/html')));
app.use(
    '/images',
    express.static(path.join(__dirname, '../frontend/images'))
);

app.use('/user/', routes);


app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`)
})