#include <pybind11/pybind11.h>
#include "gpumon/gpumon.hpp"

namespace py = pybind11;

class PyScope {
public:
    PyScope(std::string name, std::string tag) : name_(name), tag_(tag) {}

    void enter() {
        monitor_ = std::make_unique<gpumon::ScopedMonitor>(name_, tag_);
    }

    void exit(py::object exc_type, py::object exc_value, py::object traceback) {
        monitor_.reset();
    }

private:
    std::string name_;
    std::string tag_;
    std::unique_ptr<gpumon::ScopedMonitor> monitor_;
};

PYBIND11_MODULE(_gpumon_client, m) {
    m.doc() = "GPUMON Internal C++ Binding";

    m.def("init", [](std::string app_name, std::string log_path, int interval_ms) {
        gpumon::InitOptions opts;
        opts.appName = app_name;
        opts.logPath = log_path;
        opts.sampleIntervalMs = interval_ms;
        // Optional: Expose max file size to Python init if you want customizable rolling
        // opts.maxFileSizeBytes = 2 * 1024 * 1024;
        return gpumon::init(opts);
    }, py::arg("app_name"), py::arg("log_path") = "", py::arg("interval_ms") = 0);

    m.def("shutdown", &gpumon::shutdown);

    // --- ADDED THIS SECTION ---
    m.def("log_kernel", [](std::string name,
                           int gx, int gy, int gz,
                           int bx, int by, int bz,
                           long long start_ns, long long end_ns) {

        // Construct dummy CUDA types from Python integers
        dim3 grid(gx, gy, gz);
        dim3 block(bx, by, bz);

        // Empty attributes (Python doesn't have access to this low-level info)
        cudaFuncAttributes attrs = {};

        gpumon::detail::logKernelEvent(name, start_ns, end_ns, grid, block, 0, "Success", attrs);

    }, py::arg("name"),
       py::arg("gx"), py::arg("gy"), py::arg("gz"),
       py::arg("bx"), py::arg("by"), py::arg("bz"),
       py::arg("start_ns"), py::arg("end_ns"));
    // --------------------------

    py::class_<PyScope>(m, "Scope")
        .def(py::init<std::string, std::string>(), py::arg("name"), py::arg("tag") = "")
        .def("__enter__", [](PyScope &self) {
            self.enter();
            return &self;
        })
        .def("__exit__", &PyScope::exit);
}