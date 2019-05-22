using System;
using System.Drawing;
using Grasshopper.Kernel;

namespace IGA
{
    public class IGAInfo : GH_AssemblyInfo
    {
        public override string Name
        {
            get
            {
                return "IGA";
            }
        }
        public override Bitmap Icon
        {
            get
            {
                //Return a 24x24 pixel bitmap to represent this GHA library.
                return null;
            }
        }
        public override string Description
        {
            get
            {
                //Return a short string describing the purpose of this GHA library.
                return "";
            }
        }
        public override Guid Id
        {
            get
            {
                return new Guid("5b20a2ec-8b8f-4cb4-9a58-8a1555b7541c");
            }
        }

        public override string AuthorName
        {
            get
            {
                //Return a string identifying you or your company.
                return "";
            }
        }
        public override string AuthorContact
        {
            get
            {
                //Return a string representing your preferred contact details.
                return "";
            }
        }
    }
}
