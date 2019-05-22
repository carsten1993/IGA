using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace IGA
{
    class Model
    {
        private Patch patch;
        private Material material;
        //private List<PointLoad> pointLoads;
        private List<Restraint> restraints;
        private List<List<int>> res;
        private Vector<double> P1, P0, P, d;
        private Matrix<double> K_full, K;

        public Model() { }

        /*public Model(Mesh _mesh, Material _material, List<PointLoad> _pointLoads, List<Restraint> _restraints)
        {
            mesh = _mesh;
            material = _material;
            pointLoads = _pointLoads;
            restraints = _restraints;
        }*/

        public Model(Patch _patch, Material _material, List<Restraint> _restraints)
        {
            patch = _patch;
            material = _material;
            restraints = _restraints;
        }

        public Patch GetPatch() { return patch; }
        public Material GetMaterial() { return material; }
        //public List<PointLoad> GetPointLoads() { return pointLoads; }
        public List<Restraint> GetRestraints() { return restraints; }
        public List<List<int>> GetCompleteRestraints() { return res; }
        public Vector<double> GetExternalLoadVector() { return P1; }
        public Vector<double> GetSelfLoadVector() { return P0; }
        public Vector<double> GetLoadVector() { return P; }
        public Vector<double> GetDisplacementVector() { return d; }
        public Matrix<double> GetFullStiffnessMatrix() { return K_full; }
        public Matrix<double> GetStiffnessMatrix() { return K; }

        public void SetCompleteRestraints(List<List<int>> _res) { res = _res; }
        public void SetExternalLoadVector(Vector<double> _P1) { P1 = _P1; }
        public void SetSelfLoadVector(Vector<double> _P0) { P0 = _P0; }
        public void SetLoadVector(Vector<double> _P) { P = _P; }
        public void SetDisplacementVector(Vector<double> _d) { d = _d; }
        public void SetFullStiffnessMatrix(Matrix<double> _K_full) { K_full = _K_full; }
        public void SetStiffnessMatrix(Matrix<double> _K) { K = _K; }
    }
}
