import express, { urlencoded } from 'express';
import dotenv from 'dotenv';
import routes from './routes/goalRoutes.js'
import errorHandler from './middleware/errorMiddleware.js';

const PORT = process.env.PORT || 5000;

const config = dotenv.config();

const app = express()

app.use(express.json());
app.use(express.urlencoded({ extended:false }));

app.use('/api/goals/', routes);

app.use(errorHandler);

app.listen(PORT, () => console.log(`Server started on port ${PORT}`));
