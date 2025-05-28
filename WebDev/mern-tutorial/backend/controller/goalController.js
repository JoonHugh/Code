import asyncHandler from 'express-async-handler';

// @desc   Get all goals
// @route  GET /api/goals
// @access Private
export const getGoals = asyncHandler(async (req, res) => {
    res.status(200).json({ message: "Get goals" });
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
    console.log(req.body);
    res.status(200).json({ message: "Set goals" });
});

// @desc   Update goal
// @route  PUT /api/goals/:id
// @access Private
export const updateGoal = asyncHandler(async (req, res) => {
    res.status(200).json({ message: `Update goal ${req.params.id}` });
});

// @desc   Delete goal
// @route  DELETE /api/goals/:id
// @access Private
export const deleteGoal = asyncHandler(async (req, res) => {
    res.status(200).json({ message: `Delete goal ${req.params.id}` });
});