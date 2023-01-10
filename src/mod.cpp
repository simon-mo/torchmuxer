#include <pybind11/pybind11.h>
#include "fijit.h"

namespace py = pybind11;

PYBIND11_MODULE(fijit_py, m) {
    py::class_<Fijit>(m, "Fijit")
        .def(py::init<>())
        .def("run", &Fijit::run);
}