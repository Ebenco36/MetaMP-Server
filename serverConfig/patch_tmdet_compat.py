from __future__ import annotations

import sys
from pathlib import Path


def replace_exact(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new))


def replace_line_contains(path: Path, needle: str, replacement: str) -> None:
    lines = path.read_text().splitlines()
    changed = False
    new_lines: list[str] = []
    for line in lines:
        if needle in line:
            new_lines.append(replacement)
            changed = True
        else:
            new_lines.append(line)
    if changed:
        path.write_text("\n".join(new_lines) + "\n")


def remove_line_contains(path: Path, needle: str) -> None:
    lines = path.read_text().splitlines()
    filtered = [line for line in lines if needle not in line]
    if filtered != lines:
        path.write_text("\n".join(filtered) + "\n")


def patch_tree(root: Path) -> None:
    for pattern in ("*.hpp", "*.cpp"):
        for path in root.rglob(pattern):
            replace_exact(path, "#include <format>", "#include <fmt/format.h>")
            replace_exact(path, "std::format_string<", "fmt::format_string<")
            replace_exact(path, "std::format(", "fmt::format(")
            remove_line_contains(path, "#include <source_location>")

    date_hpp = root / "src/System/Date.hpp"
    replace_exact(date_hpp, "#include <sstream>", "#include <iomanip>\n#include <sstream>")
    replace_exact(
        date_hpp,
        """                std::stringstream ss;
                ss << ymd;
                return ss.str();""",
        """                std::stringstream ss;
                ss << std::setfill('0')
                   << std::setw(4) << int(ymd.year()) << '-'
                   << std::setw(2) << unsigned(ymd.month()) << '-'
                   << std::setw(2) << unsigned(ymd.day());
                return ss.str();""",
    )

    cifutil_hpp = root / "src/Utils/CifUtil.hpp"
    replace_exact(
        cifutil_hpp,
        'static constexpr std::string TMDET_MEMBRANE_ASYM_ID = "TM_";',
        'static constexpr const char* TMDET_MEMBRANE_ASYM_ID = "TM_";',
    )
    replace_exact(
        cifutil_hpp,
        'static constexpr std::string ENTRY_PREFIX = "JOB_";',
        'static constexpr const char* ENTRY_PREFIX = "JOB_";',
    )

    root_cmake = root / "CMakeLists.txt"
    replace_exact(
        root_cmake,
        "# json support\nfind_package(nlohmann_json 3.2.0 REQUIRED)\n",
        "# json support\nfind_package(nlohmann_json 3.2.0 REQUIRED)\nfind_package(fmt REQUIRED)\n",
    )

    xml_cpp = root / "src/DTOs/Xml.cpp"
    replace_exact(
        xml_cpp,
        "xmlData.chains.emplace_back(chain.id,",
        "xmlData.chains.push_back(Tmdet::VOs::XmlChain{chain.id,",
    )
    replace_exact(xml_cpp, "chain.type);", "chain.type});")

    protein_cpp = root / "src/VOs/Protein.cpp"
    replace_exact(
        protein_cpp,
        """        modifications.emplace_back(
            Tmdet::System::Date::get(),
            (std::string)"Not transmembrane protein"
        );""",
        """        modifications.push_back(Tmdet::VOs::Modification{
            Tmdet::System::Date::get(),
            (std::string)"Not transmembrane protein"
        });""",
    )

    xml_vo_cpp = root / "src/VOs/Xml.cpp"
    replace_exact(
        xml_vo_cpp,
        """        modifications.emplace_back(
            Tmdet::System::Date::get(),
            (std::string)"Not transmembrane protein"
        );""",
        """        modifications.push_back(Tmdet::VOs::Modification{
            Tmdet::System::Date::get(),
            (std::string)"Not transmembrane protein"
        });""",
    )

    reader3_cpp = root / "src/DTOs/XmlRW/Reader3.cpp"
    replace_exact(
        reader3_cpp,
        """            mods.emplace_back(mod.child(XML3_NODE_DATE).text().get(),mod.child(XML3_NODE_DESCR).text().get());""",
        """            mods.push_back(Tmdet::VOs::Modification{
                mod.child(XML3_NODE_DATE).text().get(),
                mod.child(XML3_NODE_DESCR).text().get()
            });""",
    )
    replace_exact(
        reader3_cpp,
        """            bioMatrix.matrices.emplace_back(
                matrix.attribute(XML3_ATTR_ID).as_int(),
                matrix.child(XML3_NODE_APPLY_TO_CHAIN).attribute(XML3_ATTR_CHAINID).as_string(),
                matrix.child(XML3_NODE_APPLY_TO_CHAIN).attribute(XML3_ATTR_NEW_CHAINID).as_string(),
                getTMatrix(tnode)
            );""",
        """            bioMatrix.matrices.push_back(Tmdet::VOs::Matrix{
                matrix.attribute(XML3_ATTR_ID).as_int(),
                matrix.child(XML3_NODE_APPLY_TO_CHAIN).attribute(XML3_ATTR_CHAINID).as_string(),
                matrix.child(XML3_NODE_APPLY_TO_CHAIN).attribute(XML3_ATTR_NEW_CHAINID).as_string(),
                getTMatrix(tnode)
            });""",
    )
    replace_exact(
        reader3_cpp,
        """            membranes.emplace_back(
                0.0, //todo get origo
                m_node.child(XML3_NODE_NORMAL).attribute(XML3_ATTR_Z).as_double(),
                0.0,
                10.0,
                Tmdet::Types::Membranes.at("Plain")
            );""",
        """            membranes.push_back(Tmdet::VOs::Membrane{
                0.0, //todo get origo
                m_node.child(XML3_NODE_NORMAL).attribute(XML3_ATTR_Z).as_double(),
                0.0,
                10.0,
                Tmdet::Types::Membranes.at("Plain")
            });""",
    )
    replace_exact(
        reader3_cpp,
        """            xmlChains.emplace_back(
                chainNode.attribute(XML3_ATTR_CHAINID).value(), //id
                chainNode.attribute(XML3_ATTR_CHAINID).value(), //labId
                true, //selected
                chainNode.attribute(XML3_ATTR_NUM_TM).as_int(), //numtm
                chainNode.child(XML3_NODE_SEQ).text().get(), //seq
                getRegions(chainNode), //regions
                Tmdet::Types::Chains.at(type) //type
            );""",
        """            xmlChains.push_back(Tmdet::VOs::XmlChain{
                chainNode.attribute(XML3_ATTR_CHAINID).value(), //id
                chainNode.attribute(XML3_ATTR_CHAINID).value(), //labId
                true, //selected
                chainNode.attribute(XML3_ATTR_NUM_TM).as_int(), //numtm
                chainNode.child(XML3_NODE_SEQ).text().get(), //seq
                getRegions(chainNode), //regions
                Tmdet::Types::Chains.at(type) //type
            });""",
    )

    reader4_cpp = root / "src/DTOs/XmlRW/Reader4.cpp"
    replace_exact(
        reader4_cpp,
        """            membranes.emplace_back(
                membraneNode.attribute(XML_ATTR_Z).as_double(),
                membraneNode.attribute(XML_ATTR_HALF_THICKNESS).as_double(),
                membraneNode.attribute(XML_ATTR_SPHERE_RADIUS)?
                    membraneNode.attribute(XML_ATTR_SPHERE_RADIUS).as_double():0.0),
                membraneNode.attribute(XML_ATTR_SIZE).as_double(),
                Tmdet::Types::Membranes.at(membraneNode.attribute(XML_ATTR_TYPE).as_string()
            );""",
        """            membranes.push_back(Tmdet::VOs::Membrane{
                membraneNode.attribute(XML_ATTR_Z).as_double(),
                membraneNode.attribute(XML_ATTR_HALF_THICKNESS).as_double(),
                membraneNode.attribute(XML_ATTR_SPHERE_RADIUS)
                    ? membraneNode.attribute(XML_ATTR_SPHERE_RADIUS).as_double()
                    : 0.0,
                membraneNode.attribute(XML_ATTR_SIZE).as_double(),
                Tmdet::Types::Membranes.at(membraneNode.attribute(XML_ATTR_TYPE).as_string())
            });""",
    )
    replace_exact(
        reader4_cpp,
        """            xmlChains.emplace_back(
                chainNode.attribute(XML_ATTR_AUTH_ID).as_string(),
                chainNode.attribute(XML_ATTR_LABEL_ID).as_string(),
                (type != Tmdet::Types::ChainType::NOT_SELECTED.name),
                chainNode.attribute(XML_ATTR_NUM_TM).as_int(),
                chainNode.child(XML_NODE_SEQENCE).text().get(),
                getRegions(chainNode),
                Tmdet::Types::Chains.at(type)
            );""",
        """            xmlChains.push_back(Tmdet::VOs::XmlChain{
                chainNode.attribute(XML_ATTR_AUTH_ID).as_string(),
                chainNode.attribute(XML_ATTR_LABEL_ID).as_string(),
                (type != Tmdet::Types::ChainType::NOT_SELECTED.name),
                chainNode.attribute(XML_ATTR_NUM_TM).as_int(),
                chainNode.child(XML_NODE_SEQENCE).text().get(),
                getRegions(chainNode),
                Tmdet::Types::Chains.at(type)
            });""",
    )

    neighbors_hpp = root / "src/Utils/NeighBors.hpp"
    replace_exact(
        neighbors_hpp,
        "neighbors.emplace_back(m->chain_idx,m->residue_idx);",
        "neighbors.push_back(Tmdet::VOs::CR{m->chain_idx, m->residue_idx});",
    )

    fragment_cpp = root / "src/Utils/Fragment.cpp"
    replace_exact(
        fragment_cpp,
        "ret.emplace_back(mark->chain_idx,mark->residue_idx);",
        "ret.push_back(_cr{mark->chain_idx, mark->residue_idx});",
    )

    region_handler_cpp = root / "src/Engine/RegionHandler.cpp"
    replace_line_contains(
        region_handler_cpp,
        "ret.emplace_back(beg,end-1,",
        '            ret.push_back(simpleRegion{beg, end - 1, any_cast<Tmdet::Types::Region>(chain.residues[beg].temp.at("type"))});',
    )

    symmetry_hpp = root / "src/Utils/Symmetry.hpp"
    replace_line_contains(
        symmetry_hpp,
        "bool lsqFit(",
        "                bool lsqFit(std::vector<Eigen::Vector3d>& r1, const std::vector<Eigen::Vector3d>& r2, double& rmsd, Eigen::Matrix4d& Rot) const;",
    )

    symmetry_cpp = root / "src/Utils/Symmetry.cpp"
    remove_line_contains(symmetry_cpp, "#include <span>")
    remove_line_contains(symmetry_cpp, "std::span<Eigen::Vector3d> coord1Slice")
    remove_line_contains(symmetry_cpp, "std::span<Eigen::Vector3d> coord2Slice")
    replace_exact(symmetry_cpp, "lsqFit(coord1Slice, coord2Slice, rmsd, R);", "lsqFit(coord1, coord2, rmsd, R);")
    replace_line_contains(
        symmetry_cpp,
        "bool Symmetry::lsqFit(",
        "    bool Symmetry::lsqFit(std::vector<Eigen::Vector3d>& r1, const std::vector<Eigen::Vector3d>& r2, double& rmsd, Eigen::Matrix4d& Rot) const {",
    )

    cifutil_cpp = root / "src/Utils/CifUtil.cpp"
    replace_exact(
        cifutil_cpp,
        'result.insert(insertionPosition, "," + CifUtil::TMDET_MEMBRANE_ASYM_ID);',
        'result.insert(insertionPosition, std::string(",") + CifUtil::TMDET_MEMBRANE_ASYM_ID);',
    )
    replace_exact(
        cifutil_cpp,
        'result += ("," + CifUtil::TMDET_MEMBRANE_ASYM_ID);',
        'result += (std::string(",") + CifUtil::TMDET_MEMBRANE_ASYM_ID);',
    )
    replace_exact(
        cifutil_cpp,
        "fileName = CifUtil::ENTRY_PREFIX + fileName.substr(0, 4);",
        "fileName = std::string(CifUtil::ENTRY_PREFIX) + fileName.substr(0, 4);",
    )

    src_cmake = root / "src/CMakeLists.txt"
    replace_exact(
        src_cmake,
        "TARGET_LINK_LIBRARIES(tmdet PRIVATE TmdetLib z Eigen3::Eigen gemmi::gemmi_cpp curl)",
        "TARGET_LINK_LIBRARIES(tmdet PRIVATE TmdetLib z Eigen3::Eigen gemmi::gemmi_cpp curl fmt::fmt)",
    )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_tmdet_compat.py <tmdet-root>")
    root = Path(sys.argv[1]).resolve()
    patch_tree(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
