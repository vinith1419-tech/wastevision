# TACO to Project Class Mapping

Final class IDs are fixed and must not change:

| Project ID | Project class |
|---:|---|
| 0 | plastic |
| 1 | glass |
| 2 | paper |
| 3 | metal |
| 4 | cardboard |

The preparation script maps only source categories whose labels clearly identify one of these materials. Mapping is by TACO category label, preserving the original annotated bounding box. It does not infer material from scene context.

## Included source categories

| TACO source category | Project class | Project ID |
|---|---|---:|
| Other plastic bottle | plastic | 0 |
| Clear plastic bottle | plastic | 0 |
| Plastic bottle cap | plastic | 0 |
| Disposable plastic cup | plastic | 0 |
| Other plastic cup | plastic | 0 |
| Plastic lid | plastic | 0 |
| Other plastic | plastic | 0 |
| Plastic film | plastic | 0 |
| Other plastic wrapper | plastic | 0 |
| Single-use carrier bag | plastic | 0 |
| Polypropylene bag | plastic | 0 |
| Six pack rings | plastic | 0 |
| Spread tub | plastic | 0 |
| Tupperware | plastic | 0 |
| Other plastic container | plastic | 0 |
| Plastic glooves (TACO spelling) | plastic | 0 |
| Plastic utensils | plastic | 0 |
| Plastic straw | plastic | 0 |
| Styrofoam piece | plastic | 0 |
| Glass bottle | glass | 1 |
| Broken glass | glass | 1 |
| Glass cup | glass | 1 |
| Glass jar | glass | 1 |
| Magazine paper | paper | 2 |
| Tissues | paper | 2 |
| Wrapping paper | paper | 2 |
| Normal paper | paper | 2 |
| Paper bag | paper | 2 |
| Paper straw | paper | 2 |
| Aluminium foil | metal | 3 |
| Metal bottle cap | metal | 3 |
| Food Can | metal | 3 |
| Drink can | metal | 3 |
| Metal lid | metal | 3 |
| Pop tab | metal | 3 |
| Scrap metal | metal | 3 |
| Corrugated carton | cardboard | 4 |

## Excluded TACO categories

| Source category | Reason |
|---|---|
| Aerosol | Material is not specified by the category. |
| Aluminium blister pack | Composite packaging; material identity is not a single target class. |
| Battery | Not one of the five target material categories. |
| Carded blister pack | Composite packaging; material is not unambiguous. |
| Cigarette | Not one of the five target material categories. |
| Crisp packet | Multi-layer packaging; material is ambiguous. |
| Disposable food container | Container material is not specified. |
| Drink carton | Composite carton; material is ambiguous. |
| Egg carton | Material may vary; not specified by the class. |
| Foam cup | Material is not specified by the source class. |
| Foam food container | Material is not specified by the source class. |
| Food waste | Not one of the five target material categories. |
| Garbage bag | Material is not explicitly specified. |
| Meal carton | Composite carton; material is ambiguous. |
| Other carton | Carton composition is not specified. |
| Paper cup | Often coated/composite; material is ambiguous. |
| Pizza box | The category identifies an object, not its material. |
| Plastified paper bag | Composite paper/plastic item; ambiguous target class. |
| Rope & strings | Material is not specified. |
| Shoe | Multi-material object; ambiguous target class. |
| Squeezable tube | May be plastic or composite packaging; ambiguous. |
| Toilet tube | Material is not explicit in the source label. |
| Unlabeled litter | No usable object class. |
The 23 named exclusions above correspond to the TACO categories present in the official annotation file that are not included by the mapping. If a future annotation revision adds source categories, they remain excluded until reviewed.

## License selection

The preparation script includes only images whose TACO `license` value is missing. TACO's published terms state that a missing image license defaults to CC BY 4.0. Images with explicit `CC` or OpenLitterMap ODbL values are excluded from this prepared subset to avoid mixing unclear or differing terms. Per-image source URLs and license notes are recorded in `source_manifest.csv`.
