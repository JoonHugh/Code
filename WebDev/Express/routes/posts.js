import express from 'express';
import { getPosts, getPost, createPost, putPost, deletePost } from '../controllers/postController.js';
const router = express.Router();

// middleware
const logger =  (req, res, next) => {
    console.log(`${req.method} ${req.protocol}://${req.get('host')}${req.originalUrl}`)
    next();
} // logger

// GET all posts
router.get('/', getPosts);
// GET single post
router.get('/:id', getPost);

// Create new post
router.post('/', createPost);

// Update Post
router.put('/:id', putPost);

// Delete Post
router.delete('/:id', deletePost);

export default router;