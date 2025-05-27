let ccEntries = [];

// @desc GET all ccEntries
// @route GET /
export const getAllEntries = (req, res, next) => {
  const limit = parseInt(req.query.limit);

  if (!isNaN(limit) && limit > 0) {
    return res.status(200).json(ccEntries.slice(0, limit));
  }

  res.status(200).json(ccEntries);
  // (Optional: console.log(req.query) if you meant to log the request query params)
};

// @desc GET single ccEntry
// @route GET /:name
export const getEntry = (req, res, next) => {
  const nameParam = req.params.name;

  // Look up by ccName, since that’s the key you used when pushing:
  const ccEntry = ccEntries.find((e) => e.ccName === nameParam);
  if (!ccEntry) {
    const error = new Error(`CC with name ${nameParam} doesn't exist!`);
    error.status = 404;
    return next(error);
  }

  res.status(200).json(ccEntry);
};

// @desc POST ccEntry
// @route POST /
export const postEntry = (req, res, next) => {
  const newCcEntry = {
    ccName:   req.body.name,
    ccNumber: req.body.number,
    ccMonth:  req.body.month,
    ccYear:   req.body.year,
    ccCvc:    req.body.cvc
  };

  console.log("NEW ENTRY:", newCcEntry);
  if (!newCcEntry.ccName) {
    const error = new Error("Missing an input field");
    error.status = 400;
    return next(error);
  }

  ccEntries.push(newCcEntry);
  console.log("ENTRIES:", ccEntries);
  res.status(201).json(ccEntries);
};

// @desc PUT ccEntry
// @route PUT /:name
export const putEntry = (req, res, next) => {
  const nameParam = req.params.name;
  // Again, find by .ccName:
  const ccEntry = ccEntries.find((e) => e.ccName === nameParam);
  if (!ccEntry) {
    const error = new Error(`CC with name ${nameParam} doesn't exist!`);
    error.status = 404;
    return next(error);
  }

  // Update the same keys on that object:
  ccEntry.ccName   = req.body.name;
  ccEntry.ccNumber = req.body.number;
  ccEntry.ccMonth  = req.body.month;
  ccEntry.ccYear   = req.body.year;
  ccEntry.ccCvc    = req.body.cvc;
  console.log("UPDATED:", ccEntry);

  res.status(200).json(ccEntries);
};

// @desc DELETE single ccEntry
// @route DELETE /:name
export const deleteEntry = (req, res, next) => {
  const nameParam = req.params.name;
  const ccEntry = ccEntries.find((e) => e.ccName === nameParam);
  if (!ccEntry) {
    const error = new Error(`CC with name ${nameParam} doesn't exist!`);
    error.status = 404;
    return next(error);
  }

  // Filter out entries whose ccName matches the param:
  ccEntries = ccEntries.filter((e) => e.ccName !== nameParam);
  res.status(200).json(ccEntries);
};
