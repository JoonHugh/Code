import mongoose from 'mongoose';

const connectDB = async () => {
    try {
        const conn = await mongoose.connect(process.env.MONGO_URI);
        console.log(`MongoDB Connected ${conn.connection.host}`.cyan.underline);
    } catch (error) {
        console.log('Could not connect to DB', error);
        process.exit(1);
    }
};

export default connectDB;