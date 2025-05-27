import express from 'express';
import { getAllEntries, getEntry, postEntry, putEntry, deleteEntry } from '../controllers/ccController.js'

const router = express.Router();

// GET all cc entries
router.get('/', getAllEntries);

// GET single cc entry
router.get('/:name', getEntry);

// Create cc entry
router.post('/', postEntry);

// Update cc entry
router.put('/:name', putEntry);

// Delete cc entry
router.delete('/:name', deleteEntry);

export default router;