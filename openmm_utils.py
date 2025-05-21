from openmm import unit, app, openmm


class CustomBondTabulatedForce(openmm.CustomBondForce):
    def __init__(self, energy):
        super().__init__(energy)

    def addTabulatedFunction(self, name, function):
        r"""
        addTabulatedFunction(self, name, function) -> int
        Add a tabulated function that may appear in the energy expression.

        Parameters
        ----------
        name : string
            the name of the function as it appears in expressions
        function : TabulatedFunction *
            a TabulatedFunction object defining the function. The TabulatedFunction should have been created on the heap with the "new" operator. The Force takes over ownership of it, and deletes it when the Force itself is deleted.

        Returns
        -------
        int
            the index of the function that was added
        """

        if not function.thisown:
            s = ("the %s object does not own its corresponding OpenMM object"
                 % self.__class__.__name__)
            raise Exception(s)


        val = openmm.CustomNonbondedForce_addTabulatedFunction(self, name, function)

        function.thisown=0


        return val