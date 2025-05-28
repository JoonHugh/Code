let ccEntries = [
    {
        name: "Joon",
        number: "1234567890000000",
        month: "07",
        year: "03",
        cvc: "323"
    },
    {
        "name": "Aditia",
         "number": "2222888844500040",
         "month": "02",
         "year": "01",
         "cvc": "342"
       }
];

// @desc GET all ccEntries
// @route GET /
export const getAllEntries = (req, res, next) => {
  const limit = parseInt(req.query.limit);
//   console.log("LIMIT", limit);
//   console.log("GET MY ENTRIES", ccEntries);

  if (!isNaN(limit) && limit > 0) {
    return res.status(200).json(ccEntries.slice(0, limit));
  }

  res.status(200).json(ccEntries);
  console.log(ccEntries);
  // (Optional: console.log(req.query) if you meant to log the request query params)
};

// @desc GET single ccEntry
// @route GET /:name
export const getEntry = (req, res, next) => {
  const nameParam = req.params.name;

  // Look up by name, since that’s the key you used when pushing:
  const ccEntry = ccEntries.find((e) => e.name === nameParam);
  if (!ccEntry) {
    const error = new Error(`CC with name ${nameParam} doesn't exist!`);
    error.status = 404;
    return next(error);
  }

  res.status(200).json(ccEntry);
  console.log(ccEntry);
};

// @desc POST ccEntry
// @route POST /
export const postEntry = (req, res, next) => {
    // console.log(req.query.name)
  const newCcEntry = {
    name:   req.body.name,
    number: req.body.number,
    month:  req.body.month,
    year:   req.body.year,
    cvc:    req.body.cvc
  };

//   console.log("NEW ENTRY:", newCcEntry);
  if (!newCcEntry.name) {
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
  // Again, find by .name:
  const ccEntry = ccEntries.find((e) => e.name === nameParam);
  if (!ccEntry) {
    const error = new Error(`CC with name ${nameParam} doesn't exist!`);
    error.status = 404;
    return next(error);
  }

  // Update the same keys on that object:
  if (typeof req.body.name !== 'undefined') ccEntry.name   = req.body.name;
  if (typeof req.body.number !== 'undefined') ccEntry.number = req.body.number;
  if (typeof req.body.month !== 'undefined') ccEntry.month  = req.body.month;
  if (typeof req.body.year !== 'undefined') ccEntry.year   = req.body.year;
  if (typeof req.body.cvc !== 'undefined') ccEntry.cvc    = req.body.cvc;
  
  res.status(200).json(ccEntries);
  console.log("UPDATED:", ccEntry);
};

// @desc DELETE single ccEntry
// @route DELETE /:name
export const deleteEntry = (req, res, next) => {
  const nameParam = req.params.name;
  const ccEntry = ccEntries.find((e) => e.name === nameParam);
  if (!ccEntry) {
    const error = new Error(`CC with name ${nameParam} doesn't exist!`);
    error.status = 404;
    return next(error);
  }

  // Filter out entries whose ccName matches the param:
  console.log("DELETED ENTRY", ccEntry);
  ccEntries = ccEntries.filter((e) => e.name !== nameParam);
  res.status(200).json(ccEntries);
};
