# Java 集合与相等性选型卡

## 使用时机

在 Java/Android 代码评审中，需要为集合、去重或键对象选择实现时使用。

## 决策顺序

1. 有键值对：优先 `HashMap`；需要顺序用 `LinkedHashMap`；需要排序用 `TreeMap`。
2. 仅元素去重：默认 `HashSet`；保留插入顺序用 `LinkedHashSet`；排序用 `TreeSet`。
3. 有序且读多写少：`ArrayList`；已定位位置频繁增删：`LinkedList`。
4. 自定义键或 Set 元素：同时检查 `equals()` 与 `hashCode()`。

## 红旗

- 只重写 `equals()` 或只重写 `hashCode()`。
- 把哈希碰撞当作两个对象相等。
- 为了“性能”忽略集合的顺序和重复语义。
