using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace IGA
{
    class Material
    {
        private string name;
        private double E, v, d;
        private Matrix<double> C;

        public Material()
        {

        }

        public Material(string _name, double _E, double _v, double _d, Matrix<double> _C)
        {
            name = _name;
            E = _E;
            v = _v;
            d = _d;
            C = _C;
        }

        public string GetName() { return name; }
        public double GetEmodulus() { return E; }
        public double GetPoission() { return v; }
        public double GetDensity() { return d; }
        public double GetDensityForce() { return d * 9.81 * 0.000000001; }
        public Matrix<double> GetCmatrix() { return C; }
        public string GetMaterialInfo()
        {
            return "Material: " + name +
                "\n Young's modulus: " + E.ToString() + " N/mm^2" +
                " \n Poisson's ratio: " + v.ToString() +
                " \n Density: " + d.ToString() + " kg/m^3";
        }
    }
}
