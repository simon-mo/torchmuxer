#include "fijit.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(fijit_py, m) {
  py::class_<Fijit>(m, "Fijit")
      .def(py::init<bool, bool>(), py::arg("enable_activity_api") = false,
           py::arg("enable_callback_api") = false)
      .def("run", &Fijit::run)
      .def("get_kernel_records", &Fijit::get_kernel_records, "Get kernel records");
}