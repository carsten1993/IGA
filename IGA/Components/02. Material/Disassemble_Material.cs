using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;

namespace IGA
{
    public class Disassemble_Material : GH_Component
    {
        public Disassemble_Material()
          : base("Disassemble Material", "disassemble_material",
              "Returns the properties of the given material",
              "IGA", "02. Material")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Material", "material", "The material to be disassebbled", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Info", "info", "Info about the material", GH_ParamAccess.item);
            pManager.AddTextParameter("Name", "name", "Name of the material", GH_ParamAccess.item);
            pManager.AddNumberParameter("Young's Modulus", "E", "Value of Young's Modulus for the choosen material in N/mm2 ", GH_ParamAccess.item);
            pManager.AddNumberParameter("Poission ratio", "v", "Value of poission ratio for the choosen material", GH_ParamAccess.item);
            pManager.AddNumberParameter("Density", "d", "Density the choosen material [kg/m^3]", GH_ParamAccess.item);
            pManager.AddNumberParameter("Density force", "df", "Density the choosen material [N/mm^3]", GH_ParamAccess.item);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {

            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            Material material = new Material();

            if (!DA.GetData(0, ref material)) return;

            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            string info = material.GetMaterialInfo();
            string name = material.GetName();
            double E = material.GetEmodulus();
            double v = material.GetPoission();
            double d = material.GetDensity();
            double df = material.GetDensityForce();

            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetData(0, info);
            DA.SetData(1, name);
            DA.SetData(2, E);
            DA.SetData(3, v);
            DA.SetData(4, d);
            DA.SetData(5, df);

        }

        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////


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
            get { return new Guid("0874c4aa-57c7-4352-9ebc-7aefa6fbd72e"); }
        }
    }
}