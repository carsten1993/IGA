using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;

namespace IGA
{
    public class Get_Surfaces : GH_Component
    {

        public Get_Surfaces()
          : base("Get Surfaces", "get_surfaces",
              "Returns the ID of the shape functions/control points at the surfaces for an eight cornered solid",
              "IGA", "06. Tools")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Patch", "patch", "Patch", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddIntegerParameter("Surface 1", "surface1", "Surface 1", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Surface 2", "surface2", "Surface 2", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Surface 3", "surface3", "Surface 3", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Surface 4", "surface4", "Surface 4", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Surface 5", "surface5", "Surface 5", GH_ParamAccess.list);
            pManager.AddIntegerParameter("Surface 6", "surface6", "Surface 6", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {

            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            Patch patch = new Patch();

            if (!DA.GetData(0, ref patch)) return;

            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            List<int> surface1 = patch.GetSurface1();
            List<int> surface2 = patch.GetSurface2();
            List<int> surface3 = patch.GetSurface3();
            List<int> surface4 = patch.GetSurface4();
            List<int> surface5 = patch.GetSurface5();
            List<int> surface6 = patch.GetSurface6();


            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetDataList(0, surface1);
            DA.SetDataList(1, surface2);
            DA.SetDataList(2, surface3);
            DA.SetDataList(3, surface4);
            DA.SetDataList(4, surface5);
            DA.SetDataList(5, surface6);
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
            get { return new Guid("068b069b-9c51-46db-be05-cc470d311b78"); }
        }
    }
}