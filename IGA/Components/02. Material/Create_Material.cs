using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace IGA
{
    public class Create_Material : GH_Component
    {
        public Create_Material()
          : base("Create Material", "create_material",
              "Returns a material, given the name, Young's modulus, Poisson's ratio and densityof the material",
              "IGA", "02. Material")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddTextParameter("Material name", "name", "The name of the material", GH_ParamAccess.item, "Steel S235");
            pManager.AddNumberParameter("Young's Modulus", "E", "Value of Young's Modulus for the choosen material in N/mm2 ", GH_ParamAccess.item, 210000);
            pManager.AddNumberParameter("Poission ratio", "v", "Value of poission ratio for the choosen material", GH_ParamAccess.item, 0.3);
            pManager.AddNumberParameter("Density", "d", "Density of the choosen material [kg/m^3]", GH_ParamAccess.item, 7850);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Material", "M", "Material proparties", GH_ParamAccess.item);
            pManager.AddTextParameter("Info", "info", "Info about the material", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {

            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            string name = "";
            double E = 0;
            double v = 0;
            double d = 0;

            if (!DA.GetData(0, ref name)) return;
            if (!DA.GetData(1, ref E)) return;
            if (!DA.GetData(2, ref v)) return;
            if (!DA.GetData(3, ref d)) return;

            if (E <= 0.0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Young's modulus must be bigger than zero");
                return;
            }
            if (v < 0 && v > 1)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Poisson's ratio must be between zero and one");
                return;
            }
            if (d < 0.0)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Density must be bigger than zero");
                return;
            }

            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            Matrix<double> C = CreateCMatrix(E, v);
            Material material = new Material(name, E, v, d, C);
            string info = material.GetMaterialInfo();

            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetData(0, material);
            DA.SetData(1, info);
        }

        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        Matrix<double> CreateCMatrix(double E_Y, double v_P)
        {
            double lambda = v_P * E_Y / ((1 + v_P) * (1 - 2 * v_P));
            double mu = E_Y / (2 * (1 + v_P));

            Matrix<double> C = DenseMatrix.OfArray(new double[6, 6]{
                {lambda + 2*mu, lambda, lambda, 0, 0, 0},
                {lambda, lambda + 2*mu, lambda, 0, 0, 0},
                {lambda, lambda, lambda + 2*mu, 0, 0, 0},
                {0, 0, 0, mu, 0, 0},
                {0, 0, 0, 0, mu, 0},
                {0, 0, 0, 0, 0, mu}});

            return C;
        }

        /// <summary>
        /// Provides an Icon for the component.
        /// </summary>
        protected override System.Drawing.Bitmap Icon
        {
            get
            {
                //You can add image files to your project resources and access them like this:
                // return Resources.IconForThisComponent;
                return null;
            }
        }

        /// <summary>
        /// Gets the unique ID for this component. Do not change this ID after release.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("bfbd9b43-806e-42d9-b36b-dff28a4b0b7e"); }
        }
    }
}