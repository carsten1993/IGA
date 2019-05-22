using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;

namespace IGA.Components._03._Restraints
{
    public class Disassemble_Restraint : GH_Component
    {

        public Disassemble_Restraint()
          : base("Disassemble Restraint", "disassemble_restraint",
              "Returns the properties of the given restraint",
              "IGA", "03. Restraints")
        {
        }

        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Restraint", "restraint", "The restraint to be disassebbled", GH_ParamAccess.item);
        }

        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddTextParameter("Info", "info", "Info about the restraint", GH_ParamAccess.item);
            pManager.AddIntegerParameter("ID", "id", "Returns a list of the ID of the restrained shape functions/control points", GH_ParamAccess.list);
            pManager.AddBooleanParameter("Tx", "resX", "Restraining displacement in the X-direction", GH_ParamAccess.item);
            pManager.AddBooleanParameter("Ty", "resY", "Restraining displacement in the Y-direction", GH_ParamAccess.item);
            pManager.AddBooleanParameter("Tz", "resZ", "Restraining displacement in the Z-direction", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Restraints matrix", "sortedRes", "Restraint matrix as list [id, resX, resY, resZ]", GH_ParamAccess.list);
        }

        protected override void SolveInstance(IGH_DataAccess DA)
        {

            ///////////////////////////////////////////////// INPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            Restraint restraint = new Restraint();

            if (!DA.GetData(0, ref restraint)) return;

            /////////////////////////////////////////////// FUNCTIONS //////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            string info = restraint.GetRestraintInfo();
            List<int> id = restraint.GetID();
            (bool resX, bool resY, bool resZ) = restraint.GetBooleanRestraints();
            List<List<int>> res = restraint.GetRestraints();
            List<int> sortedRes = TwoListsToList_int(res);

            //////////////////////////////////////////////// OUTPUT ////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////////////////////////

            DA.SetData(0, info);
            DA.SetDataList(1, id);
            DA.SetData(2, resX);
            DA.SetData(3, resY);
            DA.SetData(4, resZ);
            DA.SetDataList(5, sortedRes);
        }

        //////////////////////////////////////////////// METHODS ///////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        List<int> TwoListsToList_int(List<List<int>> iList)
        {
            List<int> oList = new List<int>();

            foreach (List<int> l in iList)
            {
                foreach (int d in l)
                {
                    oList.Add(d);
                }
            }

            return oList;
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
            get { return new Guid("724b07a4-9f3a-4a8a-a0ee-f2cc150fb822"); }
        }
    }
}