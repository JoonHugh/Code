import asyncHandler from 'express-async-handler';

import Goal from '../model/goalModel.js';

// @desc   Get all goals
// @route  GET /api/goals
// @access Private
export const getGoals = asyncHandler(async (req, res) => {
    const goals = await Goal.find();

    // res.status(200).json({ message: "Get goals" });

    res.status(200).json(goals);
});

// @desc   Get single goal
// @route  GET /api/goals/:id
// @access Private
export const getGoal = asyncHandler(async (req, res) => {
    res.status(200).json({ message: `Get goal ${req.params.id}` });
});

// @desc   Set goal
// @route  POST /api/goals
// @access Private
export const setGoal = asyncHandler(async (req, res) => {
    if (!req.body.text) {
        res.status(400)
        throw new Error("Please add a text field");
    }

    const goal = await Goal.create({ 
        text: req.body.text
    })
    console.log(req.body);
    // res.status(200).json({ message: "Set goals" });
    res.status(200).json(goal);

});

// @desc   Update goal
// @route  PUT /api/goals/:id
// @access Private
export const updateGoal = asyncHandler(async (req, res) => {

    const goal = await Goal.findById(req.params.id)

    if(!goal) {
        res.status(400)
        throw new Error('Goal not found');
    }

    const updatedGoal = await Goal.findByIdAndUpdate(req.params.id, req.body, {new: true});

    // res.status(200).json({ message: `Update goal ${req.params.id}` });
    res.status(200).json(updatedGoal);
});

// @desc   Delete goal
// @route  DELETE /api/goals/:id
// @access Private
export const deleteGoal = asyncHandler(async (req, res) => {
    const goal = await Goal.findByIdAndDelete(req.params.id)

    if (!goal) {
        res.status(400);
        throw new Error('Goal not found');
    }

    // res.status(200).json({ message: `Delete goal ${req.params.id}` });
    res.status(200).json({ id: req.params.id });
});