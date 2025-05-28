import mongoose from 'mongoose';

const ccSchema = mongoose.Schema({
    name: {
        type: String,
        required: [true, 'Please enter a name']
    },
    number: {
        type: Number,
        requiired: [true, 'Please enter a valid cc number']
    },
    month: {
        type: Number,
        requiired: [true, 'Please enter a valid cc month expiry date']
    },
    year: {
        type: Number,
        requiired: [true, 'Please enter a valid cc year expiry date']
    },
    cvc: {
        type: Number,
        requiired: [true, 'Please enter a valid cc security code']
    }
}, {
    timestamps: true,
});

export default mongoose.model('cc', ccSchema);